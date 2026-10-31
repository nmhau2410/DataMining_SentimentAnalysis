from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from src.utils.image_saver import save_optimized_figure
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
LABELS = [0, 1, 2]
LABEL_NAMES = ["Negative", "Neutral", "Positive"]


def load_split(filename: str):
    df = pd.read_csv(ROOT / "data" / "processed" / filename)
    expected = {"text_segmented", "sentiment"}
    if not expected.issubset(df.columns):
        raise ValueError(f"{filename} must contain {sorted(expected)}")
    return df["text_segmented"].fillna("").astype(str), df["sentiment"].astype(int)


def search_specs():
    # Settings from configs/fixed_hyperparameters.json are used as baselines.
    # The candidates below are proposed experiments, not established optima.
    tfidf = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2],
    }
    return {
        "multinomial_nb": (
            MultinomialNB(),
            {**tfidf, "model__alpha": [0.1, 1.0, 3.0], "model__fit_prior": [True, False]},
        ),
        "logistic_regression": (
            LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000, random_state=42),
            {**tfidf, "model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]},
        ),
        "linear_svm": (
            LinearSVC(loss="squared_hinge", max_iter=5000, random_state=42),
            {**tfidf, "model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]},
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Reproducible hyperparameter tuning for 3 ML models")
    parser.add_argument("--smoke", action="store_true", help="Try 1 candidate per model; NOT an optimal search")
    parser.add_argument("--only", choices=list(search_specs()), help="Run only one model")
    args = parser.parse_args()

    train_x, train_y = load_split("train.csv")
    val_x, val_y = load_split("validation.csv")
    suffix = Path("smoke") if args.smoke else Path()
    metrics_dir = ROOT / "results" / "metrics" / suffix
    cm_dir = ROOT / "results" / "confusion_matrix" / suffix
    ckpt_dir = ROOT / "checkpoints" / suffix
    for directory in [metrics_dir, cm_dir, ckpt_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    for name, (estimator, grid) in search_specs().items():
        if args.only and name != args.only:
            continue
        best_score = float("-inf")
        best_model = None
        best_params = None
        best_predictions = None
        trials = []
        candidates = list(ParameterGrid(grid))
        if args.smoke:
            candidates = candidates[:1]
        print(f"\n===== {name.upper()} : {len(candidates)} configurations =====", flush=True)
        for number, params in enumerate(candidates, start=1):
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(lowercase=True, norm="l2", token_pattern=r"(?u)\b\w+\b")),
                ("model", clone(estimator)),
            ])
            pipeline.set_params(**params)
            pipeline.fit(train_x, train_y)
            pred = pipeline.predict(val_x)
            macro_f1 = float(f1_score(val_y, pred, average="macro", zero_division=0))
            neutral_f1 = float(f1_score(val_y, pred, labels=[1], average="macro", zero_division=0))
            accuracy = float(accuracy_score(val_y, pred))
            trials.append({
                "trial": number,
                "params_json": json.dumps(params, ensure_ascii=False),
                "validation_macro_f1": macro_f1,
                "validation_neutral_f1": neutral_f1,
                "validation_accuracy": accuracy,
            })
            print(f"{number:>3}/{len(candidates)} | macro_F1={macro_f1:.4f} | neutral_F1={neutral_f1:.4f} | {params}", flush=True)
            # Tie break favors better Neutral F1, then first candidate.
            if macro_f1 > best_score:
                best_score, best_model, best_params, best_predictions = macro_f1, pipeline, params, pred
            elif macro_f1 == best_score and best_predictions is not None:
                old_neutral = f1_score(val_y, best_predictions, labels=[1], average="macro", zero_division=0)
                if neutral_f1 > old_neutral:
                    best_score, best_model, best_params, best_predictions = macro_f1, pipeline, params, pred
            pd.DataFrame(trials).to_csv(metrics_dir / f"{name}_trials.csv", index=False, encoding="utf-8-sig")

        assert best_model is not None and best_predictions is not None
        best_info = {
            "model": name,
            "selection_metric": "validation_macro_f1",
            "best_validation_macro_f1": best_score,
            "best_validation_neutral_f1": float(f1_score(val_y, best_predictions, labels=[1], average="macro", zero_division=0)),
            "best_validation_accuracy": float(accuracy_score(val_y, best_predictions)),
            "best_params": best_params,
            "trial_count": len(trials),
            "smoke_test_only": args.smoke,
            "seed": 42,
            "train_rows": len(train_y),
            "validation_rows": len(val_y),
            "test_used": False,
        }
        (metrics_dir / f"{name}_best.json").write_text(
            json.dumps(best_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (metrics_dir / f"{name}_validation_report.txt").write_text(
            classification_report(val_y, best_predictions, labels=LABELS,
                                  target_names=LABEL_NAMES, zero_division=0),
            encoding="utf-8",
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            val_y, best_predictions, labels=LABELS, display_labels=LABEL_NAMES,
            cmap="Blues", ax=ax, colorbar=False,
        )
        ax.set_title(f"{name} - Validation (not Test)")
        fig.tight_layout()
        save_optimized_figure(fig, cm_dir / f"{name}_validation.webp", dpi=180)
        plt.close(fig)
        joblib.dump(best_model, ckpt_dir / f"{name}_best_validation.joblib")
        print(f"BEST {name}: validation Macro F1 = {best_score:.4f}; parameters = {best_params}")

    print("\nSaved trial tables and model summaries to results/metrics/.")
    print("Saved VALIDATION confusion matrices to results/confusion_matrix/.")
    print("Saved fitted best pipelines to checkpoints/ (ignored by Git).")
    print("IMPORTANT: Test was not read. Run a separate final evaluation on Test after fixing params.")


if __name__ == "__main__":
    main()
