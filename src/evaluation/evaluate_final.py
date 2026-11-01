import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch
from prettytable import PrettyTable
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
CHECKPOINT_PATH = ROOT / "checkpoints" / "textcnn_final.pt"
RESULT_DIR = ROOT / "results" / "textcnn_final"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = [0, 1, 2]
LABEL_NAMES = ["Negative", "Neutral", "Positive"]


def encode_texts(texts, vocab, max_length):
    encoded = []

    for text in texts:
        tokens = str(text).lower().split()[:max_length]
        ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
        ids += [vocab["<PAD>"]] * (max_length - len(ids))
        encoded.append(ids)

    return torch.tensor(encoded, dtype=torch.long)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_sizes, dropout, num_classes=3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)

        features = []

        for conv in self.convs:
            h = torch.relu(conv(x))
            features.append(torch.max(h, dim=2).values)

        x = torch.cat(features, dim=1)
        x = self.dropout(x)

        return self.fc(x)


def main():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    vocab = checkpoint["vocab"]
    config = checkpoint["params"]

    test = pd.read_csv(DATA_DIR / "test.csv")
    test_texts = test["text_segmented"].fillna("").astype(str)

    X_test = encode_texts(test_texts, vocab, config["max_length"])
    y_test = torch.tensor(test["sentiment"].to_numpy(), dtype=torch.long)

    loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=config["embedding_dim"],
        num_filters=config["num_filters"],
        kernel_sizes=tuple(config["kernel_sizes"]),
        dropout=config["dropout"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)

            logits = model(x)
            predictions = torch.argmax(logits, dim=1)

            y_true.extend(y.tolist())
            y_pred.extend(predictions.cpu().tolist())

    accuracy = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average=None,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    metrics = {
        "model": "TextCNN",
        "test_samples": len(test),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "best_tuning_trial": checkpoint.get("best_tuning_trial"),
        "hpo_max_epochs": checkpoint.get("hpo_max_epochs"),
        "confirmation_best_epochs": checkpoint.get("confirmation_best_epochs"),
        "final_epochs": checkpoint.get("final_epochs"),
        "checkpoint": str(CHECKPOINT_PATH),
        "test_used": True,
    }

    class_report = pd.DataFrame({
        "label": LABELS,
        "class": LABEL_NAMES,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })

    predictions_df = test.copy()
    predictions_df["predicted_sentiment"] = y_pred
    predictions_df["predicted_class"] = [LABEL_NAMES[label] for label in y_pred]
    predictions_df["correct"] = predictions_df["sentiment"] == predictions_df["predicted_sentiment"]

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULT_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    class_report.to_csv(
        RESULT_DIR / "classification_report.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        cm,
        index=[f"true_{name.lower()}" for name in LABEL_NAMES],
        columns=[f"pred_{name.lower()}" for name in LABEL_NAMES],
    ).to_csv(
        RESULT_DIR / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    predictions_df.to_csv(
        RESULT_DIR / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=LABEL_NAMES,
    )

    display.plot(values_format="d")
    plt.title("TextCNN - Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    summary_table = PrettyTable()
    summary_table.field_names = ["Metric", "Value"]
    summary_table.add_row(["Accuracy", f"{accuracy:.4f}"])
    summary_table.add_row(["Macro Precision", f"{macro_precision:.4f}"])
    summary_table.add_row(["Macro Recall", f"{macro_recall:.4f}"])
    summary_table.add_row(["Macro F1", f"{macro_f1:.4f}"])

    class_table = PrettyTable()
    class_table.field_names = ["Class", "Precision", "Recall", "F1", "Support"]

    for i, name in enumerate(LABEL_NAMES):
        class_table.add_row([
            name,
            f"{precision[i]:.4f}",
            f"{recall[i]:.4f}",
            f"{f1[i]:.4f}",
            int(support[i]),
        ])

    class_table.add_row([
        "Macro Avg",
        f"{macro_precision:.4f}",
        f"{macro_recall:.4f}",
        f"{macro_f1:.4f}",
        int(sum(support)),
    ])

    cm_table = PrettyTable()
    cm_table.field_names = ["True \\ Pred"] + LABEL_NAMES

    for i, name in enumerate(LABEL_NAMES):
        cm_table.add_row([name] + cm[i].tolist())

    print("Device:", DEVICE)
    print("Test samples:", len(test))

    print("\n===== TEST RESULTS =====")
    print(summary_table)

    print("\n===== CLASSIFICATION REPORT =====")
    print(class_table)

    print("\n===== CONFUSION MATRIX =====")
    print(cm_table)

    print("\nResults:", RESULT_DIR)


if __name__ == "__main__":
    main()