import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "checkpoints" / "phobert_resume"
RESULT_DIR = ROOT / "results" / "phobert_final"
MODEL_ID = "vinai/phobert-base-v2"
LABELS = [0, 1, 2]
LABEL_NAMES = ["Negative", "Neutral", "Positive"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "labels": self.labels[index],
        }


def main():
    metadata = json.loads((CHECKPOINT_DIR / "resume_info.json").read_text(encoding="utf-8"))
    progress = json.loads((CHECKPOINT_DIR / "training_progress.json").read_text(encoding="utf-8"))
    params = metadata["params"]

    test = pd.read_csv(DATA_DIR / "test.csv")
    texts = test["text_segmented"].fillna("").astype(str).tolist()
    y_true = test["sentiment"].astype(int).tolist()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    dataset = SentimentDataset(texts, y_true, tokenizer, params["max_length"])
    loader = DataLoader(dataset, batch_size=max(1, params["batch_size"]), shuffle=False)

    config = AutoConfig.from_pretrained(
        MODEL_ID,
        num_labels=3,
        id2label={0: "Negative", 1: "Neutral", 2: "Positive"},
        label2id={"Negative": 0, "Neutral": 1, "Positive": 2},
    )
    config.hidden_dropout_prob = params["hidden_dropout_prob"]
    config.attention_probs_dropout_prob = params["attention_probs_dropout_prob"]
    config.classifier_dropout = params["classifier_dropout"]
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, config=config)
    state = torch.load(CHECKPOINT_DIR / "best_epoch.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    predictions = []
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                logits = model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                ).logits
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())

    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        y_true, predictions, labels=LABELS, average=None, zero_division=0
    )
    metrics = {
        "model": "PhoBERT",
        "test_samples": len(test),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_precision": float(precision_recall_fscore_support(y_true, predictions, labels=LABELS, average="macro", zero_division=0)[0]),
        "macro_recall": float(precision_recall_fscore_support(y_true, predictions, labels=LABELS, average="macro", zero_division=0)[1]),
        "macro_f1": float(f1_score(y_true, predictions, labels=LABELS, average="macro", zero_division=0)),
        "best_validation_macro_f1": progress["best_macro_f1"],
        "best_epoch": progress["best_epoch"],
        "checkpoint": str((CHECKPOINT_DIR / "best_epoch.pt").relative_to(ROOT)),
        "test_used": True,
    }
    report = pd.DataFrame({"label": LABELS, "class": LABEL_NAMES, "precision": precision, "recall": recall, "f1": per_class_f1, "support": support})
    predictions_df = test.copy()
    predictions_df["predicted_sentiment"] = predictions
    predictions_df["predicted_class"] = [LABEL_NAMES[label] for label in predictions]
    predictions_df["correct"] = predictions_df["sentiment"] == predictions_df["predicted_sentiment"]

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=4), encoding="utf-8")
    (RESULT_DIR / "test_report.txt").write_text(
        classification_report(y_true, predictions, labels=LABELS, target_names=LABEL_NAMES, zero_division=0),
        encoding="utf-8",
    )
    report.to_csv(RESULT_DIR / "classification_report.csv", index=False, encoding="utf-8-sig")
    predictions_df.to_csv(RESULT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(confusion_matrix(y_true, predictions, labels=LABELS), index=LABEL_NAMES, columns=LABEL_NAMES).to_csv(RESULT_DIR / "confusion_matrix.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, predictions, labels=LABELS, display_labels=LABEL_NAMES, cmap="Blues", colorbar=False, ax=ax)
    ax.set_title("PhoBERT Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
