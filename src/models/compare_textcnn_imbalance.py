import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
AUG_DIR = DATA_DIR / "augmented"
RESULT_DIR = ROOT / "results" / "textcnn_imbalance"

SEED = 42
CONFIRM_SEEDS = [123, 777, 2026]

MAX_LENGTH = 64
MIN_FREQ = 2
BATCH_SIZE = 32
MAX_EPOCHS = 50

EMBEDDING_DIM = 128
NUM_FILTERS = 100
KERNEL_SIZES = [2, 3, 4]
DROPOUT = 0.5
LR = 0.0003323701513476692
WEIGHT_DECAY = 0.0001

FOCAL_GAMMA = 2.0
CB_BETA = 0.999

STRATEGIES = {
    "original_ce": {"type": "ce", "path": DATA_DIR / "train.csv"},
    "weighted_ce": {"type": "weighted_ce", "path": DATA_DIR / "train.csv"},
    "sampler_ce": {"type": "sampler_ce", "path": DATA_DIR / "train.csv"},
    "focal": {"type": "focal", "path": DATA_DIR / "train.csv"},
    "cb_focal": {"type": "cb_focal", "path": DATA_DIR / "train.csv"},
    "aug_20": {"type": "augmentation", "path": AUG_DIR / "augmented_20.csv"},
    "aug_30": {"type": "augmentation", "path": AUG_DIR / "augmented_30.csv"},
    "aug_40": {"type": "augmentation", "path": AUG_DIR / "augmented_40.csv"},
    "aug_50": {"type": "augmentation", "path": AUG_DIR / "augmented_50.csv"},
    "aug_60": {"type": "augmentation", "path": AUG_DIR / "augmented_60.csv"}
}

CONFIRM_STRATEGIES = ["cb_focal", "weighted_ce", "sampler_ce"]


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


def encode_text(text, vocab, max_length=64):
    tokens = str(text).lower().split()[:max_length]
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    if len(ids) < max_length:
        ids += [vocab["<PAD>"]] * (max_length - len(ids))

    return ids


def encode_data(df, vocab):
    x = [encode_text(text, vocab, MAX_LENGTH) for text in df["text_segmented"]]
    y = df["sentiment"].astype(int).tolist()

    x = torch.tensor(x, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)

    return TensorDataset(x, y)


def get_class_weights(labels, device):
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    weights = len(labels) / (3 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def get_cb_weights(labels, device, beta=0.999):
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    weights = weights / weights.sum() * len(counts)

    return torch.tensor(weights, dtype=torch.float32, device=device)


def get_sampler(labels, seed):
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    class_weights = 1.0 / counts
    sample_weights = class_weights[labels]

    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
        generator=generator
    )


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        idx = torch.arange(targets.size(0), device=targets.device)
        log_pt = log_probs[idx, targets]
        pt = probs[idx, targets]

        loss = -(1 - pt).pow(self.gamma) * log_pt

        if self.alpha is not None:
            loss = self.alpha[targets] * loss

        return loss.mean()


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


def evaluate(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            pred = torch.argmax(logits, dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

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

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "negative_f1": f1[0],
        "neutral_precision": precision[1],
        "neutral_recall": recall[1],
        "neutral_f1": f1[1],
        "neutral_support": support[1],
        "positive_f1": f1[2]
    }


def get_criterion(strategy_type, labels, device):
    if strategy_type == "weighted_ce":
        weights = get_class_weights(labels, device)
        print("Class weights:", weights.cpu().numpy())
        return nn.CrossEntropyLoss(weight=weights)

    if strategy_type == "focal":
        return FocalLoss(gamma=FOCAL_GAMMA)

    if strategy_type == "cb_focal":
        weights = get_cb_weights(labels, device, CB_BETA)
        print("CB weights:", weights.cpu().numpy())
        return FocalLoss(alpha=weights, gamma=FOCAL_GAMMA)

    return nn.CrossEntropyLoss()


def get_train_loader(dataset, labels, strategy_type, seed):
    if strategy_type == "sampler_ce":
        sampler = get_sampler(labels, seed)
        return DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)


def train_one(strategy_name, config, valid, device, seed):
    print("\n" + "=" * 70)
    print("Strategy:", strategy_name)
    print("Seed:", seed)
    print("Type:", config["type"])
    print("Train:", config["path"])

    set_seed(seed)

    train = pd.read_csv(config["path"])
    labels = train["sentiment"].astype(int).to_numpy()

    print("Train samples:", len(train))
    print(train["sentiment"].value_counts().sort_index())

    vocab = build_vocab(train["text_segmented"], MIN_FREQ)

    print("Vocabulary:", len(vocab))

    train_dataset = encode_data(train, vocab)
    valid_dataset = encode_data(valid, vocab)

    train_loader = get_train_loader(train_dataset, labels, config["type"], seed)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        num_filters=NUM_FILTERS,
        kernel_sizes=KERNEL_SIZES,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = get_criterion(config["type"], labels, device)

    best_epoch = 0
    best_metrics = None
    best_macro_f1 = -1
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        total_loss = 0
        total_samples = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)

        train_loss = total_loss / total_samples
        metrics = evaluate(model, valid_loader, device)

        history.append({
            "strategy": strategy_name,
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics
        })

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Loss {train_loss:.4f} | "
            f"Acc {metrics['accuracy']:.4f} | "
            f"Macro F1 {metrics['macro_f1']:.4f} | "
            f"Neutral F1 {metrics['neutral_f1']:.4f}"
        )

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_epoch = epoch
            best_metrics = metrics.copy()

    result = {
        "strategy": strategy_name,
        "type": config["type"],
        "seed": seed,
        "best_epoch": best_epoch,
        **best_metrics
    }

    print("\nBest result:")
    print("Epoch:", best_epoch)
    print("Macro F1:", round(best_metrics["macro_f1"], 6))
    print("Neutral F1:", round(best_metrics["neutral_f1"], 6))

    del model
    torch.cuda.empty_cache()

    return result, history


def run_compare(valid, device):
    results = []
    histories = []

    for strategy_name, config in STRATEGIES.items():
        if not config["path"].exists():
            print("Skip, file not found:", config["path"])
            continue

        result, history = train_one(strategy_name, config, valid, device, SEED)
        results.append(result)
        histories.extend(history)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["macro_f1", "neutral_f1"], ascending=False).reset_index(drop=True)
    history_df = pd.DataFrame(histories)

    results_df.to_csv(RESULT_DIR / "strategy_results.csv", index=False, encoding="utf-8-sig")
    history_df.to_csv(RESULT_DIR / "training_history.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        results_df[
            [
                "strategy",
                "best_epoch",
                "accuracy",
                "macro_f1",
                "neutral_precision",
                "neutral_recall",
                "neutral_f1"
            ]
        ].to_string(index=False)
    )


def run_confirmation(valid, device):
    results = []
    histories = []

    for strategy_name in CONFIRM_STRATEGIES:
        config = STRATEGIES[strategy_name]

        for seed in CONFIRM_SEEDS:
            result, history = train_one(strategy_name, config, valid, device, seed)
            results.append(result)
            histories.extend(history)

    results_df = pd.DataFrame(results)
    history_df = pd.DataFrame(histories)

    summary = results_df.groupby("strategy").agg(
        mean_macro_f1=("macro_f1", "mean"),
        std_macro_f1=("macro_f1", "std"),
        mean_accuracy=("accuracy", "mean"),
        mean_neutral_precision=("neutral_precision", "mean"),
        mean_neutral_recall=("neutral_recall", "mean"),
        mean_neutral_f1=("neutral_f1", "mean"),
        std_neutral_f1=("neutral_f1", "std"),
        mean_best_epoch=("best_epoch", "mean")
    ).reset_index()

    summary = summary.sort_values(["mean_macro_f1", "mean_neutral_f1"], ascending=False).reset_index(drop=True)

    results_df.to_csv(RESULT_DIR / "confirmation_runs.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULT_DIR / "confirmation_results.csv", index=False, encoding="utf-8-sig")
    history_df.to_csv(RESULT_DIR / "confirmation_history.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("CONFIRMATION RESULTS")
    print("=" * 70)

    print(
        summary[
            [
                "strategy",
                "mean_macro_f1",
                "std_macro_f1",
                "mean_neutral_f1",
                "std_neutral_f1",
                "mean_neutral_recall",
                "mean_accuracy"
            ]
        ].to_string(index=False)
    )

    print("\nBest strategy:", summary.iloc[0]["strategy"])
    print("Mean Macro F1:", round(summary.iloc[0]["mean_macro_f1"], 6))
    print("Std Macro F1:", round(summary.iloc[0]["std_macro_f1"], 6))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid = pd.read_csv(DATA_DIR / "validation.csv")

    print("Device:", device)
    print("Validation samples:", len(valid))

    if args.confirm:
        print("Mode: confirmation")
        print("Seeds:", CONFIRM_SEEDS)
        run_confirmation(valid, device)
    else:
        print("Mode: comparison")
        print("Seed:", SEED)
        run_compare(valid, device)


if __name__ == "__main__":
    main()