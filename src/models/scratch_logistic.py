"""Standalone FROM-SCRATCH experiment: compare imbalance -> Optuna -> train from zero -> test.
No pretrained weights and no checkpoint loading. Run from IntelliJ project root.
Example: python -m src.models.scratch_logistic --phase tune --trials 120
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
MODEL_NAME = "logistic_regression"
TEXT_COLUMN = "text_normalized"

# ----------------------- SAME DATA AND METRICS FOR ALL FOUR MODELS -----------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_ROOT = ROOT / "results" / "scratch_valid_v3"
CHECKPOINTS_ROOT = ROOT / "checkpoints" / "scratch_valid_v3"
LABELS = [0, 1, 2]
LABEL_NAMES = ["Negative", "Neutral", "Positive"]

def read_data(text_column):
    frames=[]
    for split in ("train", "validation", "test"):
        path=DATA_DIR / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset missing: {path}")
        df=pd.read_csv(path, encoding="utf-8-sig")
        if text_column not in df.columns or "sentiment" not in df.columns:
            raise ValueError(f"{path} needs {text_column} and sentiment")
        if df[[text_column, "sentiment"]].isna().any().any():
            raise ValueError(f"Missing text/label found in {path}")
        df=df[[text_column,"sentiment"]].rename(columns={text_column:"text"}).copy()
        df["text"]=df["text"].astype(str).str.strip()
        df["sentiment"]=df["sentiment"].astype(int)
        if not set(df["sentiment"]).issubset({0,1,2}) or df["text"].eq("").any():
            raise ValueError(f"Invalid text or labels in {path}")
        if set(df["sentiment"])!={0,1,2}:
            raise ValueError(f"All three labels must be present in {path}")
        print(f"{split}: {len(df)} rows; labels={df['sentiment'].value_counts().sort_index().to_dict()}",flush=True)
        frames.append(df)
    for a,b in ((0,1),(0,2),(1,2)):
        overlap=set(frames[a].text) & set(frames[b].text)
        if overlap: raise ValueError(f"Text leakage: {len(overlap)} duplicate strings between splits {a} and {b}")
    return frames

def score(y_true,y_pred):
    p,r,f,s=precision_recall_fscore_support(y_true,y_pred,labels=LABELS,zero_division=0)
    return {"accuracy":float(accuracy_score(y_true,y_pred)),
            "macro_precision":float(p.mean()),"macro_recall":float(r.mean()),
            "macro_f1":float(f.mean()),"neutral_precision":float(p[1]),
            "neutral_recall":float(r[1]),"neutral_f1":float(f[1]),
            "per_class_f1":{LABEL_NAMES[i]:float(f[i]) for i in range(3)},
            "support":{LABEL_NAMES[i]:int(s[i]) for i in range(3)}}

def save_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")

def save_test_report(out,df,pred,extra):
    out.mkdir(parents=True,exist_ok=True)
    y=df.sentiment.to_numpy()
    m=score(y,pred)
    save_json(out / "test_metrics.json", {**m,**extra,"test_used_for_selection":False})
    (out / "classification_report.txt").write_text(
        classification_report(y,pred,labels=LABELS,target_names=LABEL_NAMES,zero_division=0),encoding="utf-8")
    pd.DataFrame(confusion_matrix(y,pred,labels=LABELS),index=LABEL_NAMES,columns=LABEL_NAMES).to_csv(
        out / "confusion_matrix.csv",encoding="utf-8-sig")
    pd.DataFrame({"text":df.text,"actual":y,"predicted":pred}).to_csv(
        out / "test_predictions.csv",index=False,encoding="utf-8-sig")
    print("TEST:", {k:round(m[k],4) for k in ("accuracy","macro_f1","neutral_f1")},flush=True)
    return m

STRATEGIES = ['original', 'balanced_weight', 'over_25', 'over_50', 'over_100', 'under_4x', 'hybrid_50', 'smote']
SEED = 42
BASE = {'ngram_max': 2, 'min_df': 2, 'max_df': 1., 'max_features': 20000,
        'sublinear_tf': True, 'C': 1.0, 'alpha': 1., 'fit_prior': True}


def vectorizer(p):
    return TfidfVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b',
                           ngram_range=(1, int(p['ngram_max'])), min_df=int(p['min_df']),
                           max_df=float(p['max_df']), max_features=int(p['max_features']),
                           sublinear_tf=bool(p['sublinear_tf']), dtype=np.float32)


def estimator(name, p, strategy):
    if name != MODEL_NAME: raise ValueError(f"Unexpected model: {name}")
    return LogisticRegression(C=float(p['C']), class_weight='balanced' if strategy=='balanced_weight' else None,
                              random_state=SEED, solver='lbfgs', max_iter=2000)


def re_sample_texts(texts, labels, method, seed):
    """Oversampling/undersampling happens on the current training partition ONLY."""
    x = np.asarray(texts, dtype=object); y = np.asarray(labels, dtype=int)
    if method in ('original', 'balanced_weight', 'smote'):
        return x, y
    rng = np.random.default_rng(seed)
    counts = np.bincount(y, minlength=3)
    majority, minority = int(counts.max()), int(counts.min())
    if minority < 1:
        raise ValueError('A training fold is missing one of the sentiment labels')
    if method.startswith('over_'):
        ratio = {'over_25':0.25, 'over_50':0.50, 'over_100':1.0}[method]
        targets = [max(int(c), int(np.ceil(ratio * majority))) for c in counts]
    elif method == 'under_4x':
        targets = [min(int(c), 4 * minority) for c in counts]
    elif method == 'hybrid_50':
        # downsample majority to 75%, upsample minority to 50% of original majority
        targets = [min(int(c), int(np.ceil(majority * .75))) if c > majority*.75
                   else max(int(c), int(np.ceil(majority * .50))) for c in counts]
    else:
        raise ValueError(method)
    ids = np.concatenate([rng.choice(np.where(y == label)[0], size=n,
                                     replace=n > counts[label]) for label, n in enumerate(targets)])
    rng.shuffle(ids)
    return x[ids], y[ids]


def fit_fresh(name, method, params, texts, labels, seed):
    """New vectorizer and new sklearn estimator for EVERY single fit."""
    texts, labels = re_sample_texts(texts, labels, method, seed)
    tfidf = vectorizer(params)
    x_vec = tfidf.fit_transform(texts)
    if method == 'smote':
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError as exc:
            raise RuntimeError('SMOTE requires: pip install imbalanced-learn') from exc
        # Feature interpolation (not text augmentation). Apply after TF-IDF fit, inside train/fold only.
        x_vec, labels = SMOTE(random_state=seed, k_neighbors=3).fit_resample(x_vec, labels)
    clf = estimator(name, params, method)
    if name == 'multinomial_nb' and method == 'balanced_weight':
        counts = np.bincount(labels, minlength=3).astype(float)
        sample_weight = len(labels)/(3*counts[labels])
        clf.fit(x_vec, labels, sample_weight=sample_weight)
    else:
        clf.fit(x_vec, labels)
    return Pipeline([('tfidf', tfidf), ('model', clf)])


def compare(name, train, val, seeds):
    out = RESULTS_ROOT/name; out.mkdir(parents=True, exist_ok=True)
    runs = []
    for method in STRATEGIES:
        for seed in seeds:
            p = fit_fresh(name, method, BASE, train.text, train.sentiment, seed)
            metrics = score(val.sentiment, p.predict(val.text))
            runs.append({'method':method, 'seed':seed, **{k:metrics[k] for k in
                ['accuracy', 'macro_f1', 'neutral_f1', 'neutral_precision', 'neutral_recall']}})
            print(f'{name:21} {method:16} seed={seed}: val macro F1={metrics["macro_f1"]:.4f} neutral F1={metrics["neutral_f1"]:.4f}', flush=True)
    pd.DataFrame(runs).to_csv(out/'strategy_runs.csv', index=False, encoding='utf-8-sig')
    summary = (pd.DataFrame(runs).groupby('method',as_index=False)
        .agg(mean_macro_f1=('macro_f1','mean'), std_macro_f1=('macro_f1','std'),
             mean_neutral_f1=('neutral_f1','mean'), mean_accuracy=('accuracy','mean'))
        .sort_values(['mean_macro_f1','mean_neutral_f1'],ascending=False).reset_index(drop=True))
    summary.to_csv(out/'strategy_comparison.csv', index=False, encoding='utf-8-sig')
    # The unbalanced original is a REPORTING baseline, not eligible for final selection.
    chosen = str(summary.loc[summary.method != 'original'].iloc[0].method)
    save_json(out/'strategy_selection.json', {'model':name,'method':chosen,'seeds':list(seeds),
                                              'selection_metric':'mean_validation_macro_f1','test_used':False})
    print(f'CHOSEN for {name}: {chosen}',flush=True)
    return chosen


def tune(name, method, train, valid, trials):
    """Each Optuna trial fits on train.csv; objective = Macro F1 on validation.csv.

    --trials is the TOTAL target for this study, including previously saved trials.
    No CV, internal holdout or test observations enter model selection.
    """
    out = RESULTS_ROOT/name
    out.mkdir(parents=True, exist_ok=True)
    study_path = out/f'optuna_{method}.db'
    study = optuna.create_study(study_name=f'{name}_{method}_official_validation_v3',
        direction='maximize', storage=f'sqlite:///{study_path.resolve().as_posix()}',
        load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED))
    settings={'model':name, 'strategy':method, 'selection':'official_validation_macro_f1',
              'train':'data/processed/train.csv', 'valid':'data/processed/validation.csv',
              'seed':SEED, 'text_column':TEXT_COLUMN}
    if study.trials and study.user_attrs.get('settings')!=settings:
        raise RuntimeError('Existing study settings differ. Back up and rename the v3 Optuna DB before changing settings.')
    study.set_user_attr('settings',settings)

    def objective(trial):
        p = dict(BASE)
        p.update({'ngram_max':trial.suggest_categorical('ngram_max',[1,2,3]),
                  'min_df':trial.suggest_categorical('min_df',[1,2,3,5]),
                  'max_df':trial.suggest_categorical('max_df',[0.9,1.]),
                  'max_features':trial.suggest_categorical('max_features',[10000,20000,30000]),
                  'sublinear_tf':trial.suggest_categorical('sublinear_tf',[True,False])})
        if name == 'multinomial_nb':
            p['alpha'] = trial.suggest_float('alpha',1e-4,10.,log=True)
            p['fit_prior'] = trial.suggest_categorical('fit_prior',[True,False])
        else:
            p['C'] = trial.suggest_float('C',1e-3,100.,log=True)
        model = fit_fresh(name,method,p,train.text,train.sentiment,SEED)
        val_score = score(valid.sentiment,model.predict(valid.text))
        trial.set_user_attr('validation_neutral_f1',val_score['neutral_f1'])
        trial.set_user_attr('validation_accuracy',val_score['accuracy'])
        return float(val_score['macro_f1'])

    done = len(study.trials)
    remaining = max(0,trials-done)
    print(f'{name}: already {done} trials; target {trials}; running {remaining} more.',flush=True)
    if remaining:
        study.optimize(objective,n_trials=remaining,show_progress_bar=False,gc_after_trial=True)
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        raise RuntimeError('No Optuna trial completed successfully; inspect logs before final training.')
    study.trials_dataframe().to_csv(out/'optuna_trials.csv',index=False,encoding='utf-8-sig')
    best = dict(BASE); best.update(study.best_params)
    info = {'model':name,'method':method,'best_params':best,
            'validation_macro_f1':float(study.best_value),
            'validation_neutral_f1':float(study.best_trial.user_attrs['validation_neutral_f1']),
            'selection_data':'data/processed/validation.csv',
            'train_data':'data/processed/train.csv',
            'best_trial':int(study.best_trial.number),'completed_trials':len(study.trials),
            'test_used':False}
    save_json(out/'optuna_best.json',info)
    print(f'{name} Optuna BEST official VALID macro F1: {study.best_value:.4f}',flush=True)
    return info


def final(name, method, params, train, val, test, text_column):
    # Fresh TF-IDF and freshly initialized estimator. Does NOT load old .joblib models.
    trainval=pd.concat([train,val],ignore_index=True)
    model=fit_fresh(name,method,params,trainval.text,trainval.sentiment,SEED)
    out=RESULTS_ROOT/name
    ckpt=CHECKPOINTS_ROOT/f"{name}.joblib"
    ckpt.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,ckpt,compress=3)
    save_test_report(out,test,model.predict(test.text),{
        "model":name,"imbalance_method":method,"best_params":params,
        "trainval_rows":len(trainval),"text_column":text_column,
        "weights_pretrained":False,"checkpoint":str(ckpt)})
    print("Saved checkpoint:",ckpt,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase",choices=["compare","tune","final","all"],default="all")
    ap.add_argument("--trials",type=int,default=120, help="Target TOTAL trials in saved v3 study")
    ap.add_argument("--seeds",type=int,nargs="+",default=[42,123,777])
    ap.add_argument("--text-column",choices=["text_normalized","text_segmented"],default="text_normalized",
        help="Use text_normalized to avoid an external pretrained word-segmentation model")
    a=ap.parse_args()
    if a.trials<1 or not a.seeds: ap.error("Requires trials>=1 and at least one seed")
    global TEXT_COLUMN
    TEXT_COLUMN=a.text_column
    train,val,test=read_data(a.text_column)
    name=MODEL_NAME
    out=RESULTS_ROOT/name
    if a.phase in ("compare","all"):
        method=compare(name,train,val,a.seeds)
        sel=json.loads((out/"strategy_selection.json").read_text(encoding="utf-8"))
        sel["text_column"]=a.text_column
        save_json(out/"strategy_selection.json",sel)
    else:
        p=out/"strategy_selection.json"
        if not p.is_file(): raise FileNotFoundError(f"Run --phase compare first: {p}")
        sel=json.loads(p.read_text(encoding="utf-8"))
        if sel.get("text_column")!=a.text_column:raise ValueError("Text column differs from compare run")
        method=sel["method"]
    if a.phase in ("tune","all"):
        info=tune(name,method,train,val,a.trials)
        info["text_column"]=a.text_column
        save_json(out/"optuna_best.json",info)
    elif a.phase=="final":
        p=out/"optuna_best.json"
        if not p.is_file():raise FileNotFoundError(f"Run --phase tune first: {p}")
        info=json.loads(p.read_text(encoding="utf-8"))
        if info["method"]!=method or info.get("text_column")!=a.text_column:
            raise ValueError("Selected method or text column has changed; run tuning again")
    if a.phase in ("final","all"):
        final(name,method,info["best_params"],train,val,test,a.text_column)

if __name__=="__main__":
    main()
