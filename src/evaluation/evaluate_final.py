from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
CHECKPOINT_PATH = ROOT / "checkpoints" / "textcnn_final_weighted.pt"
RESULT_DIR = ROOT / "results" / "textcnn_final_weighted"

BATCH_SIZE = 128
LABEL_NAMES = ["Negative", "Neutral", "Positive"]


def encode_text(text, vocab, max_length):
    tokens = str(text).lower().split()[:max_length]
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    if len(ids) < max_length:
        ids += [vocab["<PAD>"]] * (max_length - len(ids))

    return ids


def encode_data(df, vocab, max_length):
    x = [encode_text(text, vocab, max_length) for text in df["text_segmented"]]
    y = df["sentiment"].astype(int).tolist()

    x = torch.tensor(x, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)

    return TensorDataset(x, y)


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
        x = x.permute(0, 2, 1)

        conv_outs = []

        for conv in self.convs:
            out = torch.relu(conv(x))
            out = torch.max(out, dim=2).values
            conv_outs.append(out)

        x = torch.cat(conv_outs, dim=1)
        x = self.dropout(x)

        return self.fc(x)


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Checkpoint:", CHECKPOINT_PATH)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    vocab = checkpoint["vocab"]
    params = checkpoint["params"]
    max_length = checkpoint["max_length"]

    print("Trial:", checkpoint["trial"])
    print("Strategy:", checkpoint["strategy"])
    print("Vocabulary:", len(vocab))
    print("Max length:", max_length)
    print("Final epochs:", checkpoint["final_epochs"])

    test = pd.read_csv(DATA_DIR / "test.csv")

    print("\nTest samples:", len(test))
    print(test["sentiment"].value_counts().sort_index())

    test_dataset = encode_data(test, vocab, max_length)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=params["embedding_dim"],
        num_filters=params["num_filters"],
        kernel_sizes=params["kernel_sizes"],
        dropout=params["dropout"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true = []
    y_pred = []
    probabilities = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)

            y_true.extend(y.numpy())
            y_pred.extend(pred.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average=None,
        zero_division=0
    )

    print("\n" + "=" * 70)
    print("TEXTCNN FINAL TEST RESULTS")
    print("=" * 70)

    print("Accuracy:", round(accuracy, 6))
    print("Macro Precision:", round(macro_precision, 6))
    print("Macro Recall:", round(macro_recall, 6))
    print("Macro F1:", round(macro_f1, 6))

    print("\nPer-class results:")

    for i, name in enumerate(LABEL_NAMES):
        print(
            f"{name:8s} | "
            f"Precision {precision[i]:.6f} | "
            f"Recall {recall[i]:.6f} | "
            f"F1 {f1[i]:.6f} | "
            f"Support {support[i]}"
        )

    metrics = {
        "model": "TextCNN",
        "trial": checkpoint["trial"],
        "strategy": checkpoint["strategy"],
        "test_samples": len(test),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "negative_precision": float(precision[0]),
        "negative_recall": float(recall[0]),
        "negative_f1": float(f1[0]),
        "neutral_precision": float(precision[1]),
        "neutral_recall": float(recall[1]),
        "neutral_f1": float(f1[1]),
        "positive_precision": float(precision[2]),
        "positive_recall": float(recall[2]),
        "positive_f1": float(f1[2])
    }

    pd.DataFrame([metrics]).to_json(
        RESULT_DIR / "test_metrics.json",
        orient="records",
        indent=4,
        force_ascii=False
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        RESULT_DIR / "classification_report.csv",
        encoding="utf-8-sig"
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    cm_df = pd.DataFrame(
        cm,
        index=LABEL_NAMES,
        columns=LABEL_NAMES
    )

    cm_df.to_csv(
        RESULT_DIR / "confusion_matrix.csv",
        encoding="utf-8-sig"
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticklabels(LABEL_NAMES)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("TextCNN Confusion Matrix")

    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    predictions = test.copy()

    predictions["true_label"] = y_true
    predictions["true_name"] = [LABEL_NAMES[i] for i in y_true]
    predictions["predicted_label"] = y_pred
    predictions["predicted_name"] = [LABEL_NAMES[i] for i in y_pred]

    probabilities = torch.tensor(probabilities)

    predictions["prob_negative"] = probabilities[:, 0].numpy()
    predictions["prob_neutral"] = probabilities[:, 1].numpy()
    predictions["prob_positive"] = probabilities[:, 2].numpy()
    predictions["confidence"] = probabilities.max(dim=1).values.numpy()
    predictions["correct"] = predictions["true_label"] == predictions["predicted_label"]

    predictions.to_csv(
        RESULT_DIR / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nConfusion Matrix:")
    print(cm_df)

    print("\nSaved:")
    print(RESULT_DIR / "test_metrics.json")
    print(RESULT_DIR / "classification_report.csv")
    print(RESULT_DIR / "confusion_matrix.csv")
    print(RESULT_DIR / "confusion_matrix.png")
    print(RESULT_DIR / "test_predictions.csv")


if __name__ == "__main__":
    main()