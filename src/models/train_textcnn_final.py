import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
CONFIG_PATH = ROOT / "results" / "textcnn_tuning_weighted" / "selected_config.json"
RESULT_DIR = ROOT / "results" / "textcnn_final_weighted"
CHECKPOINT_DIR = ROOT / "checkpoints"

SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_vocab(texts, min_freq=2):
    counter = Counter()

    for text in texts:
        counter.update(str(text).lower().split())

    vocab = {"<PAD>": 0, "<UNK>": 1}

    for token, count in counter.items():
        if count >= min_freq:
            vocab[token] = len(vocab)

    return vocab


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


def get_class_weights(labels, device):
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    weights = len(labels) / (3 * counts)

    return torch.tensor(weights, dtype=torch.float32, device=device)


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
    set_seed(SEED)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        selected = json.load(f)

    params = selected["params"]
    max_length = selected["fixed"]["max_length"]
    min_freq = selected["fixed"]["min_freq"]
    final_epochs = selected["final_epochs"]

    print("Selected trial:", selected["trial"])
    print("Strategy:", selected["strategy"])
    print("Final epochs:", final_epochs)
    print("Params:", params)

    train = pd.read_csv(DATA_DIR / "train.csv")
    valid = pd.read_csv(DATA_DIR / "validation.csv")

    final_train = pd.concat([train, valid], ignore_index=True)

    print("\nFinal train samples:", len(final_train))
    print(final_train["sentiment"].value_counts().sort_index())

    vocab = build_vocab(final_train["text_segmented"], min_freq)

    print("Vocabulary:", len(vocab))

    dataset = encode_data(final_train, vocab, max_length)
    labels = final_train["sentiment"].astype(int).to_numpy()

    generator = torch.Generator()
    generator.manual_seed(SEED)

    loader = DataLoader(
        dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        generator=generator
    )

    class_weights = get_class_weights(labels, device)

    print("Class weights:", class_weights.cpu().numpy())

    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=params["embedding_dim"],
        num_filters=params["num_filters"],
        kernel_sizes=params["kernel_sizes"],
        dropout=params["dropout"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"]
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = []

    print("\nTraining final model...")

    for epoch in range(1, final_epochs + 1):
        model.train()

        total_loss = 0
        total_samples = 0
        correct = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            pred = torch.argmax(logits, dim=1)

            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
            correct += (pred == y).sum().item()

        train_loss = total_loss / total_samples
        train_accuracy = correct / total_samples

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy
        })

        print(
            f"Epoch {epoch:02d}/{final_epochs} | "
            f"Loss {train_loss:.4f} | "
            f"Accuracy {train_accuracy:.4f}"
        )

    checkpoint_path = CHECKPOINT_DIR / "textcnn_final_weighted.pt"

    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "params": params,
        "max_length": max_length,
        "min_freq": min_freq,
        "final_epochs": final_epochs,
        "strategy": "weighted_ce",
        "class_weights": class_weights.cpu().tolist(),
        "seed": SEED,
        "trial": selected["trial"]
    }, checkpoint_path)

    pd.DataFrame(history).to_csv(
        RESULT_DIR / "training_history.csv",
        index=False,
        encoding="utf-8-sig"
    )

    counts = final_train["sentiment"].value_counts().sort_index()

    summary = {
        "trial": selected["trial"],
        "strategy": "weighted_ce",
        "seed": SEED,
        "train_samples": len(final_train),
        "negative": int(counts.get(0, 0)),
        "neutral": int(counts.get(1, 0)),
        "positive": int(counts.get(2, 0)),
        "vocab_size": len(vocab),
        "max_length": max_length,
        "min_freq": min_freq,
        "final_epochs": final_epochs,
        "embedding_dim": params["embedding_dim"],
        "num_filters": params["num_filters"],
        "kernel_sizes": params["kernel_sizes"],
        "dropout": params["dropout"],
        "lr": params["lr"],
        "weight_decay": params["weight_decay"],
        "batch_size": params["batch_size"],
        "class_weights": class_weights.cpu().tolist(),
        "optimizer": "AdamW",
        "loss": "WeightedCrossEntropy",
        "checkpoint": str(checkpoint_path),
        "test_used": False
    }

    with open(RESULT_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print("\n" + "=" * 70)
    print("FINAL TRAINING COMPLETE")
    print("=" * 70)

    print("Trial:", selected["trial"])
    print("Strategy: weighted_ce")
    print("Train + Validation:", len(final_train))
    print("Vocabulary:", len(vocab))
    print("Final epochs:", final_epochs)
    print("Checkpoint:", checkpoint_path)
    print("Test used: False")

    print("\nSaved:")
    print(RESULT_DIR / "training_history.csv")
    print(RESULT_DIR / "training_summary.json")
    print(checkpoint_path)


if __name__ == "__main__":
    main()