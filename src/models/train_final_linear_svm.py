import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from src.models.tune_ml_optuna import ROOT, load_split, make_pipeline

MODEL_NAME = "linear_svm"
LABELS = [0, 1, 2]
LABEL_NAMES = ["Negative", "Neutral", "Positive"]


def main():
    data_dir = ROOT / "data" / "processed"
    required_files = [
        data_dir / f"{split}.csv"
        for split in ("train", "validation", "test")
    ]

    missing = [
        str(path) for path in required_files
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing preprocessed datasets:\n" + "\n".join(missing)
        )

    best_json = ROOT / "results" / "optuna" / MODEL_NAME / "best.json"

    if not best_json.is_file():
        raise FileNotFoundError(
            f"Optuna best parameters not found: {best_json}"
        )

    best_result = json.loads(
        best_json.read_text(encoding="utf-8")
    )

    if best_result.get("model") != MODEL_NAME:
        raise ValueError(
            f"Unexpected model: {best_result.get('model')}"
        )

    params = best_result["best_params"]

    train_x, train_y = load_split("train.csv")
    val_x, val_y = load_split("validation.csv")
    test_x, test_y = load_split("test.csv")

    trainval_x = pd.concat(
        [train_x, val_x], ignore_index=True
    )
    trainval_y = pd.concat(
        [train_y, val_y], ignore_index=True
    )

    print(f"Model: {MODEL_NAME}")
    print(f"Best Optuna trial: {best_result.get('best_trial')}")
    print(f"Train: {len(train_x)}")
    print(f"Validation: {len(val_x)}")
    print(f"Final Train: {len(trainval_x)}")
    print(f"Test: {len(test_x)}")
    print(f"Best Parameters: {params}")

    model = make_pipeline(MODEL_NAME, params)
    model.fit(trainval_x, trainval_y)

    checkpoint_path = (
            ROOT / "checkpoints" / f"{MODEL_NAME}_final.joblib"
    )

    checkpoint_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    joblib.dump(model, checkpoint_path, compress=3)

    predictions = model.predict(test_x)

    metrics = {
        "model": MODEL_NAME,
        "best_trial": best_result.get("best_trial"),
        "train_rows": len(train_x),
        "validation_rows": len(val_x),
        "final_train_rows": len(trainval_x),
        "test_rows": len(test_x),
        "accuracy": float(
            accuracy_score(test_y, predictions)
        ),
        "precision_macro": float(
            precision_score(
                test_y, predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                test_y, predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                test_y, predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_neutral": float(
            f1_score(
                test_y, predictions,
                labels=[1],
                average="macro",
                zero_division=0,
            )
        ),
        "best_params": params,
    }

    output_dir = ROOT / "results" / f"{MODEL_NAME}_final"
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "test_metrics.json").write_text(
        json.dumps(
            metrics, ensure_ascii=False, indent=4
        ),
        encoding="utf-8",
    )

    report = classification_report(
        test_y,
        predictions,
        labels=LABELS,
        target_names=LABEL_NAMES,
        zero_division=0,
    )

    (output_dir / "test_report.txt").write_text(
        report, encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        test_y,
        predictions,
        labels=LABELS,
        display_labels=LABEL_NAMES,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )

    ax.set_title("Linear SVM - Test Confusion Matrix")
    fig.tight_layout()

    fig.savefig(
        output_dir / "test_confusion_matrix.png",
        dpi=160,
        )

    plt.close(fig)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print(f"Neutral F1: {metrics['f1_neutral']:.4f}")
    print(report)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()