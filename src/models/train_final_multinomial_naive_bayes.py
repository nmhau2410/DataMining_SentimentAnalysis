import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay,accuracy_score,classification_report, f1_score,precision_score,recall_score,)
from src.models.tune_ml_optuna import ROOT, load_split, make_pipeline
MODEL_NAME = "multinomial_nb"
LABELS = [0, 1, 2]
LABEL_NAMES = ["Negative", "Neutral", "Positive"]
def main():
    data_dir = ROOT / "data" / "processed"
    needed_files = [data_dir / f"{split}.csv" for split in ("train", "validation", "test")]
    missing = [str(file) for file in needed_files if not file.is_file()]
    if missing:
        raise FileNotFoundError("Thiếu dữ liệu đã tiền xử lý:\n")
    best_json = ROOT / "results" / "optuna" / MODEL_NAME / "best.json"
    if not best_json.is_file():
        raise FileNotFoundError(f"Không tìm thấy bộ tham số: {best_json}")
    best_result = json.loads(best_json.read_text(encoding="utf-8"))
    params = best_result["best_params"]
    train_x, train_y = load_split("train.csv")
    val_x, val_y = load_split("validation.csv")
    test_x, test_y = load_split("test.csv")
    trainval_x = pd.concat([train_x, val_x], ignore_index=True)
    trainval_y = pd.concat([train_y, val_y], ignore_index=True)
    model = make_pipeline(MODEL_NAME, params)
    print("{MODEL_NAME.upper()}")
    print("Bộ siêu tham số:", params)
    model.fit(trainval_x, trainval_y)
    predictions = model.predict(test_x)
    metrics = {
        "model": MODEL_NAME,
        "train_rows": len(train_x),
        "validation_rows": len(val_x),
        "final_train_rows": len(trainval_x),
        "test_rows": len(test_x),
        "accuracy": float(accuracy_score(test_y, predictions)),
        "precision_macro": float(precision_score(test_y, predictions, labels=LABELS, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(test_y, predictions, labels=LABELS, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(test_y, predictions, labels=LABELS, average="macro", zero_division=0)),
        "f1_neutral": float(f1_score(test_y, predictions, labels=[1], average="macro", zero_division=0)),
        "best_params": params,
    }
    out_dir = ROOT / "results" / f"{MODEL_NAME}_final"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=4), encoding="utf-8" )
    report = classification_report(test_y, predictions,labels=LABELS,target_names=LABEL_NAMES,zero_division=0,)
    (out_dir / "test_report.txt").write_text(report, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(test_y, predictions,labels=LABELS,display_labels=LABEL_NAMES,cmap="Blues",colorbar=False,ax=ax,)
    ax.set_title(f"{MODEL_NAME} - Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "test_confusion_matrix.png", dpi=160)
    plt.close(fig)
    checkpoint_path = ROOT / "checkpoints" / f"{MODEL_NAME}_final.joblib"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, checkpoint_path, compress=3)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print(f"F1 Neutral: {metrics['f1_neutral']:.4f}")
    print(report)
    print("Mô hình đã lưu:", checkpoint_path)
    print("Kết quả:", out_dir)
if __name__ == "__main__":
    main()
