"""RoBERTa / PhoBERT-style FROM SCRATCH: no pretrained weights/tokenizer.
Selection: fit train.csv, choose hyperparameters and best epoch on validation.csv.
Suggested GPU RTX 2050 commands (from IntelliJ project root):
  python -m src.models.scratch_phobert --phase compare --arch compact --screen-epochs 6
  python -m src.models.scratch_phobert --phase tune --arch compact --trials 120 --tune-epochs 50
  python -m src.models.scratch_phobert --phase final --arch compact
This is NOT official pretrained VinAI PhoBERT. Compact is NOT PhoBERT-base.
"""
from __future__ import annotations
import argparse
import gc
import json
import math
import random
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

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

SEED = 42
TEXT_COLUMN = "text_normalized"
# Note: BPE vocabulary size is a fixed experimental setting, trained fresh on each training split.
STRATEGIES = ['original', 'weighted_ce', 'weighted_sampler', 'focal',
              'class_balanced_focal', 'over_25', 'over_50', 'over_100']
SPECIAL = ['<s>', '<pad>', '</s>', '<unk>', '<mask>']


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_tokenizer(texts, vocab_size):
    """Create a brand-new BPE tokenizer from ONLY the provided training text."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
    from transformers import PreTrainedTokenizerFast
    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # learns BPE on the project text column
    trainer = trainers.BpeTrainer(vocab_size=int(vocab_size), min_frequency=2,
                                  special_tokens=SPECIAL, show_progress=False)
    tokenizer.train_from_iterator((str(t) for t in texts),trainer=trainer,length=len(texts))
    tokenizer.post_processor = processors.TemplateProcessing(
        single='<s> $A </s>', pair='<s> $A </s> </s> $B </s>',
        special_tokens=[('<s>', tokenizer.token_to_id('<s>')), ('</s>', tokenizer.token_to_id('</s>'))])
    hf = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token='<s>', cls_token='<s>',
                                 eos_token='</s>', sep_token='</s>',pad_token='<pad>',
                                 unk_token='<unk>', mask_token='<mask>')
    return hf


def build_model(tokenizer, arch, max_length, dropout):
    """Both config and RoBERTa weights are created locally. No from_pretrained()."""
    from transformers import RobertaConfig, RobertaForSequenceClassification
    dims = (12, 768, 12, 3072) if arch=='base' else (4, 256, 4, 1024)
    layers, hidden, heads, intermediate = dims
    config = RobertaConfig(vocab_size=len(tokenizer), hidden_size=hidden,
                           num_hidden_layers=layers, num_attention_heads=heads,
                           intermediate_size=intermediate, max_position_embeddings=max_length+2,
                           type_vocab_size=1, pad_token_id=tokenizer.pad_token_id,
                           bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                           num_labels=3, hidden_dropout_prob=dropout,
                           attention_probs_dropout_prob=dropout,
                           classifier_dropout=dropout, id2label={0:'Negative',1:'Neutral',2:'Positive'},
                           label2id={'Negative':0,'Neutral':1,'Positive':2})
    model = RobertaForSequenceClassification(config)  # random-init model
    # Memory reduction without changing weights or introducing any pretraining.
    if hasattr(model,'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    return model


def choose_rows(df, method, seed):
    """Over-sampling is row duplication only, on the training partition."""
    if method not in ('over_25','over_50','over_100'):
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    ratios = {'over_25':.25, 'over_50':.50, 'over_100':1.0}
    counts = df.sentiment.value_counts(); target = max(counts)
    indices = []
    for label in range(3):
        found = np.flatnonzero(df.sentiment.to_numpy()==label)
        n = max(len(found),math.ceil(target*ratios[method]))
        indices.extend(rng.choice(found,size=n,replace=n>len(found)).tolist())
    rng.shuffle(indices)
    return df.iloc[indices].reset_index(drop=True)


def loader_for(df, tokenizer, max_length, batch_size, seed, method='original', train=False):
    enc = tokenizer(df.text.astype(str).tolist(), truncation=True,
                    padding='max_length', max_length=max_length, return_tensors='pt')
    labels = torch.tensor(df.sentiment.to_numpy(), dtype=torch.long)
    dataset = TensorDataset(enc['input_ids'], enc['attention_mask'], labels)
    if train and method=='weighted_sampler':
        counts = np.bincount(labels.numpy(),minlength=3).astype(float)
        weights = 1.0/counts[labels.numpy()]
        gen = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(torch.tensor(weights,dtype=torch.double),
                                        num_samples=len(labels),replacement=True,generator=gen)
        return DataLoader(dataset,batch_size=batch_size,sampler=sampler,num_workers=0)
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(dataset,batch_size=batch_size,shuffle=train,generator=gen,num_workers=0)


def class_weights(y, effective=False, beta=0.999):
    c = np.bincount(np.asarray(y,dtype=int),minlength=3).astype(float)
    if (c==0).any():raise ValueError('All three classes must be represented in training')
    if effective:
        w = (1-beta)/np.maximum(1-np.power(beta,c),1e-9)
        w = w/w.mean()
    else:
        w = len(y)/(3*c)
        w = w/w.mean()
    return w.astype(np.float32)


def compute_loss(logits, labels, method, weights, gamma):
    if method=='weighted_ce':
        return F.cross_entropy(logits.float(),labels,weight=weights)
    if method in ('focal','class_balanced_focal'):
        log_probs = F.log_softmax(logits.float(),dim=-1)
        picked = log_probs.gather(dim=-1,index=labels.unsqueeze(-1)).squeeze(-1)
        per_sample = -((1-picked.exp())**gamma)*picked
        if method=='class_balanced_focal':per_sample = per_sample*weights[labels]
        return per_sample.mean()
    return F.cross_entropy(logits.float(), labels)


def test_or_validate(model,loader,device):
    model.eval()
    yt,yp=[],[]
    with torch.inference_mode():
        for ids,mask,labels in loader:
            ids=ids.to(device);mask=mask.to(device)
            with torch.autocast('cuda',dtype=torch.float16,enabled=device.type=='cuda'):
                logits=model(input_ids=ids,attention_mask=mask).logits
            yt.extend(labels.numpy().tolist());yp.extend(logits.argmax(-1).cpu().tolist())
    return score(yt,yp),yt,yp


def fit_from_zero(train,valid,method,params,arch,seed,epochs,patience,vocab_size,save_checkpoint=None,eval_test=None):
    """Creates a NEW tokenizer + NEW random weights on every call; NO optimizer/model restoration."""
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_aug = choose_rows(train,method,seed)
    tokenizer = train_tokenizer(train.text.astype(str).tolist(),vocab_size)
    model = build_model(tokenizer,arch,int(params['max_length']),float(params['dropout'])).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  scratch {arch} | {method} | {n_params:,} randomly initialized parameters',flush=True)
    batch=int(params['batch_size']);accum=int(params['grad_accumulation'])
    train_loader = loader_for(train_aug,tokenizer,int(params['max_length']),batch,seed,method,True)
    valid_loader = (loader_for(valid,tokenizer,int(params['max_length']),max(4,batch),seed)
                    if valid is not None else None)
    optimizer = torch.optim.AdamW(model.parameters(),lr=float(params['learning_rate']),
                                  weight_decay=float(params['weight_decay']))
    total_steps=max(1,math.ceil(len(train_loader)/accum)*epochs)
    warmup_steps=int(float(params['warmup_ratio'])*total_steps)
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    scaler = torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    weights = torch.tensor(class_weights(train_aug.sentiment,
                           effective=method=='class_balanced_focal'),dtype=torch.float32,device=device)
    best_score=-1.;best_epoch=0;best_metrics=None;history=[];stale=0
    try:
        for epoch in range(1,epochs+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            losses=[]
            for step,(ids,mask,labels) in enumerate(train_loader):
                ids=ids.to(device);mask=mask.to(device);labels=labels.to(device)
                with torch.autocast('cuda',dtype=torch.float16,enabled=device.type=='cuda'):
                    logits=model(input_ids=ids,attention_mask=mask).logits
                    loss=compute_loss(logits,labels,method,weights,float(params['focal_gamma']))
                losses.append(float(loss.detach().cpu()))
                # Correct average for the last incomplete accumulation group.
                group_start=(step//accum)*accum
                group_n=min(accum,len(train_loader)-group_start)
                scaler.scale(loss/group_n).backward()
                if ((step+1)%accum==0) or step==len(train_loader)-1:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    prev_scale = scaler.get_scale()
                    scaler.step(optimizer);scaler.update()
                    if scaler.get_scale()>=prev_scale:scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            record={'epoch':epoch,'train_loss':float(np.mean(losses))}
            if valid_loader is not None:
                val_metrics,_,_=test_or_validate(model,valid_loader,device)
                record.update({'validation_macro_f1':val_metrics['macro_f1'],
                               'validation_neutral_f1':val_metrics['neutral_f1']})
                if val_metrics['macro_f1']>best_score+1e-7:
                    best_score=val_metrics['macro_f1'];best_epoch=epoch
                    best_metrics=val_metrics;stale=0
                else:
                    stale+=1
                print(f'    epoch {epoch}/{epochs}: val macro F1={val_metrics["macro_f1"]:.4f} neutral F1={val_metrics["neutral_f1"]:.4f}',flush=True)
            else:
                best_epoch=epoch
                print(f'    epoch {epoch}/{epochs}: training loss={record["train_loss"]:.4f}',flush=True)
            history.append(record)
            if valid_loader is not None and stale>=patience:break
        test_metrics=None
        if save_checkpoint is not None:
            save_checkpoint.mkdir(parents=True,exist_ok=True)
            model.save_pretrained(save_checkpoint,safe_serialization=True)
            tokenizer.save_pretrained(save_checkpoint)
            save_json(save_checkpoint/'scratch_provenance.json',{
                'pretrained_weights':False,'pretrained_tokenizer':False,
                'weights_initialization':'RobertaForSequenceClassification(RobertaConfig(...))',
                'tokenizer_training_data':'train+validation only',
                'architecture':arch,'method':method,'model_params':n_params,
                'tokenizer_vocab_size':len(tokenizer),'max_length':int(params['max_length']),
                'epochs':epochs, 'optimizer':'AdamW','experiment_params':params})
        if eval_test is not None:
            loader=loader_for(eval_test,tokenizer,int(params['max_length']),max(4,batch),seed)
            test_metrics,yt,yp=test_or_validate(model,loader,device)
            # test metrics are saved by the caller; never used in method or hyperparameter selection
        return {'best_epoch':best_epoch,'best_val_f1':best_score,'best_metrics':best_metrics,
                'history':history,'test_predictions':yp if eval_test is not None else None,
                'model_params':n_params}
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError('CUDA out of memory. Try --arch compact, or reduce batch_size and max_length.') from exc
    finally:
        del model,optimizer,train_loader,valid_loader,tokenizer,scaler,scheduler
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()


def defaults(arch):
    return {'learning_rate':3e-4,'weight_decay':0.01,'batch_size':2 if arch=='base' else 8,
            'grad_accumulation':8 if arch=='base' else 2,'max_length':96,
            'dropout':0.1,'warmup_ratio':0.08,'focal_gamma':2.0}


def output_dir(arch):
    return RESULTS_ROOT/('phobert_style_base_random_init' if arch=='base' else 'roberta_compact_random_init')


def compare(train,valid,arch,epochs,seeds,vocab_size):
    out = output_dir(arch);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for method in STRATEGIES:
        for seed in seeds:
            run=fit_from_zero(train,valid,method,defaults(arch),arch,seed,
                              epochs,patience=3,vocab_size=vocab_size)
            m=run['best_metrics']
            rows.append({'method':method,'seed':seed,'val_macro_f1':m['macro_f1'],
                         'val_neutral_f1':m['neutral_f1'],'val_accuracy':m['accuracy'],
                         'best_epoch':run['best_epoch']})
            pd.DataFrame(rows).to_csv(out/'strategy_runs.csv',index=False,encoding='utf-8-sig')
    summary = (pd.DataFrame(rows).groupby('method',as_index=False)
        .agg(mean_macro_f1=('val_macro_f1','mean'),std_macro_f1=('val_macro_f1','std'),
             mean_neutral_f1=('val_neutral_f1','mean'))
        .sort_values(['mean_macro_f1','mean_neutral_f1'],ascending=False).reset_index(drop=True))
    summary.to_csv(out/'strategy_comparison.csv',index=False,encoding='utf-8-sig')
    # Original CE is only the reporting baseline: choose an actual imbalance treatment.
    chosen=str(summary.loc[summary.method != 'original'].iloc[0].method)
    save_json(out/'strategy_selection.json',{'method':chosen,'arch':arch,
        'selection_metric':'mean_validation_macro_f1','seeds':seeds,'vocab_size':vocab_size,
        'text_column':TEXT_COLUMN,'test_used':False})
    print('BEST strategy:',chosen,flush=True)
    return chosen


def tune(train,valid,method,arch,trials,epochs,vocab_size):
    """Optuna trains on train.csv and selects params/epoch using official validation.csv."""
    out = output_dir(arch);out.mkdir(parents=True,exist_ok=True)
    study_path = out / ('optuna_' + method + '.db')
    study = optuna.create_study(direction='maximize',
        study_name=f'phobert_random_{arch}_{method}_official_validation_v3',
        storage=f"sqlite:///{study_path.resolve().as_posix()}",
        load_if_exists=True,sampler=optuna.samplers.TPESampler(seed=SEED))
    settings={'arch':arch,'method':method,'tune_epochs':epochs,'vocab_size':vocab_size,
              'data': 'train.csv for fit; validation.csv for Optuna selection', 'text_column':TEXT_COLUMN}
    if study.trials and study.user_attrs.get('settings') != settings:
        raise RuntimeError('This Optuna study has different settings. Archive local study DB before changing them.')
    study.set_user_attr('settings',settings)
    def objective(trial):
        p=defaults(arch)
        p.update({'learning_rate':trial.suggest_float('learning_rate',1e-4,8e-4,log=True),
                  'weight_decay':trial.suggest_float('weight_decay',0.,0.1),
                  'batch_size':trial.suggest_categorical('batch_size',[2,4] if arch=='base' else [4,8,16]),
                  'grad_accumulation':trial.suggest_categorical('grad_accumulation',[2,4,8]),
                  'max_length':trial.suggest_categorical('max_length',[64,96,128]),
                  'dropout':trial.suggest_float('dropout',0.05,0.3),
                  'warmup_ratio':trial.suggest_float('warmup_ratio',0.,.15),
                  'focal_gamma':trial.suggest_float('focal_gamma',1.,3.) if 'focal' in method else 2.0})
        try:
            report=fit_from_zero(train,valid,method,p,arch,SEED+trial.number,
                                 epochs,patience=10,vocab_size=vocab_size)
        except RuntimeError as exc:
            if 'CUDA out of memory' in str(exc):
                raise optuna.TrialPruned(str(exc)) from exc
            raise
        trial.set_user_attr('best_epoch',report['best_epoch'])
        trial.set_user_attr('validation_neutral_f1',report['best_metrics']['neutral_f1'])
        trial.set_user_attr('epoch_history',report['history'])
        return float(report['best_val_f1'])
    done=len(study.trials)
    remaining=max(0,trials-done)
    print(f'PhoBERT-style: already {done} trials; target {trials}; running {remaining} more.',flush=True)
    if remaining:
        study.optimize(objective,n_trials=remaining,show_progress_bar=False,gc_after_trial=True)
    if not any(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials):
        raise RuntimeError('All PhoBERT trials failed/pruned; lower memory requirements or change architecture.')
    study.trials_dataframe().to_csv(out/'optuna_trials.csv',index=False,encoding='utf-8-sig')
    history=study.best_trial.user_attrs.get('epoch_history')
    if history:
        pd.DataFrame(history).to_csv(out/'optuna_best_epoch_history.csv',index=False,encoding='utf-8-sig')
    p=defaults(arch);p.update(study.best_params)
    info={'method':method,'arch':arch,'best_params':p,'vocab_size':vocab_size,
          'best_epoch':int(study.best_trial.user_attrs['best_epoch']),
          'best_validation_macro_f1':float(study.best_value),
          'best_trial':int(study.best_trial.number),
          'tuning_split':'official validation.csv (train.csv ONLY for fitting)',
          'completed_trials':len(study.trials),
          'pretrained':False,'text_column':TEXT_COLUMN,'test_used':False}
    save_json(out/'optuna_best.json',info)
    return info


def final(train,valid,test,info,arch):
    """Randomly reinitialize BOTH tokenizer and network; fit train+validation, test once."""
    trainval=pd.concat([train,valid],ignore_index=True)
    epochs=max(1,int(info['best_epoch']))
    out=output_dir(arch)
    checkpoint=CHECKPOINTS_ROOT/('phobert_style_base_random_init' if arch=='base' else 'roberta_compact_random_init')
    run=fit_from_zero(trainval,None,info['method'],info['best_params'],arch,SEED,
                      epochs,patience=epochs+1,vocab_size=info['vocab_size'],
                      save_checkpoint=checkpoint,eval_test=test)
    pd.DataFrame(run['history']).to_csv(out/'final_training_history.csv',index=False,encoding='utf-8-sig')
    save_test_report(out,test,run['test_predictions'],
              {'model': 'PhoBERT-style architecture with random weights' if arch=='base' else 'Compact RoBERTa with random weights',
               'strategy':info['method'],'params':info['best_params'],
               'final_epochs':epochs,'train_val_rows':len(trainval),
               'text_column':TEXT_COLUMN,'random_initialized_weights':True,'scratch_BPE_tokenizer':True,
               'checkpoint':str(checkpoint),'test_used_for_tuning':False})
    print(f'Saved trained model to {checkpoint}',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=['all','compare','tune','final'],default='all')
    ap.add_argument('--arch',choices=['base','compact'],default='base')
    ap.add_argument('--trials',type=int,default=120,help='Target TOTAL trials in saved v3 study')
    ap.add_argument('--screen-epochs',type=int,default=6)
    ap.add_argument('--tune-epochs',type=int,default=50)
    ap.add_argument('--seeds',nargs='+',type=int,default=[42])
    ap.add_argument('--vocab-size',type=int,default=8000)
    ap.add_argument('--allow-cpu',action='store_true')
    ap.add_argument('--text-column',choices=['text_normalized','text_segmented'],default='text_normalized')
    a=ap.parse_args()
    if min(a.trials,a.screen_epochs,a.tune_epochs,a.vocab_size)<1:ap.error('All numeric settings must be positive')
    if not torch.cuda.is_available() and not a.allow_cpu:
        ap.error('GPU CUDA not available. Use NVIDIA PyTorch CUDA build, or --allow-cpu (very slow).')
    print('Device:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
    print('Strict scratch: tokenizer trained from data; ALL model weights randomized; NO pretrained weights.')
    global TEXT_COLUMN
    TEXT_COLUMN=a.text_column
    train,valid,test=read_data(a.text_column)
    out=output_dir(a.arch)
    if a.phase in ('all','compare'):
        method=compare(train,valid,a.arch,a.screen_epochs,a.seeds,a.vocab_size)
    else:
        import json
        sel=out/'strategy_selection.json'
        if not sel.is_file():raise FileNotFoundError(f'Run --phase compare first: {sel}')
        chosen=json.loads(sel.read_text(encoding='utf-8'))
        if chosen['arch']!=a.arch or chosen['vocab_size']!=a.vocab_size or chosen.get('text_column')!=a.text_column:
            raise ValueError('Architecture or vocab size changed since comparison')
        method=chosen['method']
    if a.phase in ('all','tune'):
        info=tune(train,valid,method,a.arch,a.trials,a.tune_epochs,a.vocab_size)
    elif a.phase=='final':
        import json
        best=out/'optuna_best.json'
        if not best.is_file():raise FileNotFoundError(f'Run --phase tune first: {best}')
        info=json.loads(best.read_text(encoding='utf-8'))
        if info['method']!=method or info['arch']!=a.arch or info.get('text_column')!=a.text_column:
            raise ValueError('Mismatch of chosen method or architecture; rerun Optuna tune')
    if a.phase in ('all','final'):
        final(train,valid,test,info,a.arch)

if __name__=='__main__':
    main()
