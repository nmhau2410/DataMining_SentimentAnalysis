import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
CONFIG_PATH = ROOT / "configs" / "selected_hyperparameters.json"
TUNING_DIR = ROOT / "results" / "textcnn_tuning_v2"
BEST_INFO_PATH = TUNING_DIR / "textcnn_best.json"
CONFIRM_PATH = TUNING_DIR / "confirmation_trials.csv"
RESULT_DIR = ROOT / "results" / "textcnn_final"
CHECKPOINT_PATH = ROOT / "checkpoints" / "textcnn_final.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_vocab(texts, min_freq):
    counter = Counter()

    for text in texts:
        counter.update(str(text).lower().split())

    vocab = {"<PAD>": 0, "<UNK>": 1}

    for word in sorted(counter):
        if counter[word] >= min_freq:
            vocab[word] = len(vocab)

    return vocab


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


def load_selected_config():
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    if "textcnn" not in data:
        raise KeyError("Không tìm thấy config TextCNN đã chọn.")

    return data["textcnn"]


def get_tuning_info():
    best_info = json.loads(BEST_INFO_PATH.read_text(encoding="utf-8"))
    best_trial = int(best_info["best_trial"])

    confirmation = pd.read_csv(CONFIRM_PATH)
    rows = confirmation[confirmation["trial"] == best_trial]

    if rows.empty:
        raise ValueError(f"Không tìm thấy confirmation của trial {best_trial}.")

    best_epochs = rows["best_epoch"].to_numpy()
    median_best_epoch = int(np.median(best_epochs))

    return best_trial, best_epochs.tolist(), median_best_epoch


def make_criterion(labels, class_weight):
    if class_weight == "none":
        return nn.CrossEntropyLoss(), None

    counts = np.bincount(labels.numpy(), minlength=3)
    weights = len(labels) / (3 * np.maximum(counts, 1))
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    return nn.CrossEntropyLoss(weight=weight_tensor), weights.tolist()


def main():
    config = load_selected_config()
    best_trial, confirmation_epochs, final_epochs = get_tuning_info()

    set_seed(SEED)

    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "validation.csv")

    final_train = pd.concat([train, val], ignore_index=True)

    texts = final_train["text_segmented"].fillna("").astype(str)
    labels = torch.tensor(final_train["sentiment"].to_numpy(), dtype=torch.long)

    vocab = build_vocab(texts, config["min_freq"])
    X = encode_texts(texts, vocab, config["max_length"])

    generator = torch.Generator()
    generator.manual_seed(SEED)

    loader = DataLoader(
        TensorDataset(X, labels),
        batch_size=config["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=config["embedding_dim"],
        num_filters=config["num_filters"],
        kernel_sizes=tuple(config["kernel_sizes"]),
        dropout=config["dropout"],
    ).to(DEVICE)

    if config["optimizer"] != "AdamW":
        raise ValueError("Final TextCNN hiện chỉ hỗ trợ AdamW.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    criterion, class_weights = make_criterion(labels, config["class_weight"])
    history = []

    print("Device:", DEVICE)
    print("Original Train:", len(train))
    print("Original Validation:", len(val))
    print("Final Train:", len(final_train))
    print("Vocabulary:", len(vocab))
    print("Best tuning trial:", best_trial)
    print("HPO max epochs:", config["epochs"])
    print("Confirmation best epochs:", confirmation_epochs)
    print("Final epochs:", final_epochs)
    print("Class weights:", class_weights)

    print("\nConfig:")
    print(json.dumps(config, ensure_ascii=False, indent=4))

    for epoch in range(1, final_epochs + 1):
        model.train()

        total_loss = 0.0
        total_items = 0

        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            total_items += len(y)

        train_loss = total_loss / total_items

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
        })

        print(f"Epoch {epoch:02d}/{final_epochs} | Train Loss={train_loss:.4f}", flush=True)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(history).to_csv(
        RESULT_DIR / "training_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "model": "TextCNN",
        "training_data": "train + validation",
        "train_rows_before_merge": len(train),
        "validation_rows_before_merge": len(val),
        "final_train_rows": len(final_train),
        "seed": SEED,
        "best_tuning_trial": best_trial,
        "hpo_max_epochs": config["epochs"],
        "confirmation_best_epochs": confirmation_epochs,
        "final_epochs": final_epochs,
        "epoch_rule": "median_best_epoch_from_confirmation",
        "vocab_size": len(vocab),
        "class_weights": class_weights,
        "params": config,
        "test_used": False,
    }

    with open(RESULT_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "params": config,
            "seed": SEED,
            "best_tuning_trial": best_trial,
            "hpo_max_epochs": config["epochs"],
            "confirmation_best_epochs": confirmation_epochs,
            "final_epochs": final_epochs,
            "train_rows": len(final_train),
        },
        CHECKPOINT_PATH,
    )

    print("\n===== COMPLETED =====")
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Results:", RESULT_DIR)
    print("Test dataset was NOT used.")


if __name__ == "__main__":
    main()