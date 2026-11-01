
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.utils.image_saver import save_optimized_figure


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
RESULT_DIR = ROOT / "results" / "optuna"
CHECKPOINT_DIR = ROOT / "checkpoints"

MODELS = [
    "multinomial_nb",
    "logistic_regression",
    "linear_svm",
]

SEED = 42
CV_FOLDS = 3


def load_split(filename):
    df = pd.read_csv(DATA_DIR / filename)

    required = {"text_segmented", "sentiment"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"{filename} is missing required columns: {required}"
        )

    x = df["text_segmented"].fillna("").astype(str)
    y = df["sentiment"].astype(int)

    return x, y


def suggest_params(trial, model_name):
    if model_name == "multinomial_nb":
        feature_options = [None, 10000, 30000]
    else:
        feature_options = [10000, 20000, 30000]

    params = {
        "ngram_range": trial.suggest_categorical(
            "ngram_range",
            ["1,1", "1,2", "1,3"],
        ),
        "min_df": trial.suggest_categorical(
            "min_df",
            [1, 2, 3, 5],
        ),
        "max_df": trial.suggest_categorical(
            "max_df",
            [0.9, 1.0],
        ),
        "sublinear_tf": trial.suggest_categorical(
            "sublinear_tf",
            [True, False],
        ),
        "max_features": trial.suggest_categorical(
            "max_features",
            feature_options,
        ),
    }

    if model_name == "multinomial_nb":
        params["alpha"] = trial.suggest_float(
            "alpha",
            1e-4,
            100.0,
            log=True,
        )

        params["fit_prior"] = trial.suggest_categorical(
            "fit_prior",
            [True, False],
        )

    elif model_name == "logistic_regression":
        params["C"] = trial.suggest_float(
            "C",
            1e-3,
            100.0,
            log=True,
        )

        params["class_weight"] = trial.suggest_categorical(
            "class_weight",
            [None, "balanced"],
        )

    elif model_name == "linear_svm":
        params["C"] = trial.suggest_float(
            "C",
            1e-3,
            100.0,
            log=True,
        )

        params["class_weight"] = trial.suggest_categorical(
            "class_weight",
            [None, "balanced"],
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return params


def make_pipeline(model_name, params):
    ngram_range = tuple(
        int(value)
        for value in params["ngram_range"].split(",")
    )

    tfidf = TfidfVectorizer(
        lowercase=True,
        norm="l2",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=ngram_range,
        min_df=params["min_df"],
        max_df=params["max_df"],
        sublinear_tf=params["sublinear_tf"],
        max_features=params["max_features"],
    )

    if model_name == "multinomial_nb":
        model = MultinomialNB(
            alpha=params["alpha"],
            fit_prior=params["fit_prior"],
        )

    elif model_name == "logistic_regression":
        model = LogisticRegression(
            C=params["C"],
            l1_ratio=0.0,
            solver="lbfgs",
            class_weight=params["class_weight"],
            max_iter=1500,
            tol=1e-3,
            random_state=SEED,
        )

    elif model_name == "linear_svm":
        model = LinearSVC(
            C=params["C"],
            loss="squared_hinge",
            class_weight=params["class_weight"],
            dual="auto",
            max_iter=3000,
            tol=1e-3,
            random_state=SEED,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([
        ("tfidf", tfidf),
        ("model", model),
    ])


def validate_study(study, model_name):
    if model_name == "multinomial_nb":
        return

    old_params = {"penalty", "solver", "loss"}

    for trial in study.trials:
        if old_params.intersection(trial.params):
            raise RuntimeError(
                f"Study {study.study_name} contains trials "
                "from the previous search space. "
                f"Stop training and remove the old database "
                f"at results/optuna/{model_name}/study.db "
                "before starting the new search."
            )


def run_model(
        model_name,
        train_x,
        train_y,
        val_x,
        val_y,
        n_trials,
        jobs,
):
    model_dir = RESULT_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    db_path = model_dir / "study.db"
    storage_url = f"sqlite:///{db_path.as_posix()}"

    study_name = f"{model_name}_optuna_v1"

    startup_trials = (
        15 if model_name == "multinomial_nb"
        else 10
    )

    sampler = optuna.samplers.TPESampler(
        seed=SEED,
        n_startup_trials=startup_trials,
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage_url,
        load_if_exists=True,
    )

    validate_study(study, model_name)

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    def objective(trial):
        params = suggest_params(trial, model_name)

        pipeline = make_pipeline(
            model_name,
            params,
        )

        scores = cross_val_score(
            pipeline,
            train_x,
            train_y,
            scoring="f1_macro",
            cv=cv,
            n_jobs=jobs,
            error_score="raise",
        )

        trial.set_user_attr(
            "cv_std",
            float(scores.std()),
        )

        return float(scores.mean())

    print(
        f"\n========== {model_name.upper()} ==========",
        flush=True,
    )

    print("Study:", study_name, flush=True)
    print("Existing trials:", len(study.trials), flush=True)
    print("Additional trials:", n_trials, flush=True)
    print("CV folds:", CV_FOLDS, flush=True)
    print("CV jobs:", jobs, flush=True)

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=1,
            gc_after_trial=True,
        )

    except KeyboardInterrupt:
        print(
            "\nTraining interrupted. Completed trials were saved.",
            flush=True,
        )

    study.trials_dataframe().to_csv(
        model_dir / "trials.csv",
        index=False,
        encoding="utf-8-sig",
        )

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed_trials:
        print("No completed trials found.")
        return

    best_trial = study.best_trial
    best_params = best_trial.params.copy()

    best_pipeline = make_pipeline(
        model_name,
        best_params,
    )

    best_pipeline.fit(
        train_x,
        train_y,
    )

    predictions = best_pipeline.predict(val_x)

    val_macro_f1 = float(
        f1_score(
            val_y,
            predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
    )

    val_neutral_f1 = float(
        f1_score(
            val_y,
            predictions,
            labels=[1],
            average="macro",
            zero_division=0,
        )
    )

    val_accuracy = float(
        accuracy_score(
            val_y,
            predictions,
        )
    )

    summary = {
        "model": model_name,
        "optimizer": "Optuna TPE",
        "study_name": study_name,
        "selection_metric": "train_cv_macro_f1",
        "cv_folds": CV_FOLDS,
        "best_cv_macro_f1": float(best_trial.value),
        "best_cv_std": best_trial.user_attrs.get(
            "cv_std"
        ),
        "validation_macro_f1": val_macro_f1,
        "validation_neutral_f1": val_neutral_f1,
        "validation_accuracy": val_accuracy,
        "best_params": best_params,
        "best_trial": best_trial.number,
        "completed_trials": len(completed_trials),
        "seed": SEED,
        "train_rows": len(train_x),
        "validation_rows": len(val_x),
        "test_used": False,
    }

    with open(
            model_dir / "best.json",
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=4,
        )

    report = classification_report(
        val_y,
        predictions,
        labels=[0, 1, 2],
        target_names=[
            "Negative",
            "Neutral",
            "Positive",
        ],
        zero_division=0,
    )

    (model_dir / "validation_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        val_y,
        predictions,
        labels=[0, 1, 2],
        display_labels=[
            "Negative",
            "Neutral",
            "Positive",
        ],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )

    ax.set_title(
        f"{model_name} - Validation Confusion Matrix"
    )

    fig.tight_layout()

    save_optimized_figure(
        fig,
        model_dir / "confusion_matrix.webp",
        dpi=160,
        )

    plt.close(fig)

    checkpoint_path = (
            CHECKPOINT_DIR
            / f"{model_name}_best_validation.joblib"
    )

    joblib.dump(
        best_pipeline,
        checkpoint_path,
        compress=3,
    )

    print("\n========== FINAL RESULT ==========")
    print("Model:", model_name)
    print("Best CV Macro F1:", best_trial.value)
    print("Validation Macro F1:", val_macro_f1)
    print("Validation Neutral F1:", val_neutral_f1)
    print("Validation Accuracy:", val_accuracy)
    print("Best Parameters:", best_params)
    print("Best Trial:", best_trial.number)
    print("Completed Trials:", len(completed_trials))
    print("Checkpoint:", checkpoint_path)
    print("Results:", model_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--only",
        choices=["all"] + MODELS,
        default="all",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    if args.trials < 1:
        parser.error("--trials must be >= 1")

    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    train_x, train_y = load_split("train.csv")
    val_x, val_y = load_split("validation.csv")

    print("Train:", len(train_x))
    print("Validation:", len(val_x))
    print("Test dataset is not used.")

    selected_models = (
        MODELS
        if args.only == "all"
        else [args.only]
    )

    for model_name in selected_models:
        run_model(
            model_name=model_name,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            n_trials=args.trials,
            jobs=args.jobs,
        )


if __name__ == "__main__":
    main()
