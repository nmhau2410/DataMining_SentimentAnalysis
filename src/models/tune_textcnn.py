import argparse
import copy
import json
import random
from collections import Counter
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from src.utils.image_saver import save_optimized_figure
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 1. CAU HINH CHUNG

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"

LABEL_NAMES = ["Negative", "Neutral", "Positive"]
KERNEL_SIZES = (3, 4, 5)

DEVICE = torch.device("cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# 2. DOC DU LIEU

def load_data(smoke=False):
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "validation.csv")

    if smoke:
        train = train.sample(
            n=min(512, len(train)), random_state=42
        )
        val = val.sample(
            n=min(256, len(val)), random_state=42
        )

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)

    print("Train:", len(train))
    print("Validation:", len(val))

    return train, val


# 3. TAO VOCABULARY

def build_vocab(texts, min_freq=2):
    counter = Counter()

    for text in texts:
        counter.update(str(text).lower().split())

    vocab = {"<PAD>": 0, "<UNK>": 1}

    for word in sorted(counter):
        if counter[word] >= min_freq:
            vocab[word] = len(vocab)

    return vocab


def encode_texts(texts, vocab, max_length=64):
    encoded = []

    for text in texts:
        tokens = str(text).lower().split()

        ids = [
            vocab.get(token, vocab["<UNK>"])
            for token in tokens[:max_length]
        ]

        ids += [0] * (max_length - len(ids))
        encoded.append(ids)

    return torch.tensor(encoded, dtype=torch.long)


# 4. TEXTCNN MODEL

class TextCNN(nn.Module):
    def __init__(
            self,
            vocab_size,
            embedding_dim=128,
            num_filters=100,
            kernel_sizes=(3, 4, 5),
            dropout=0.5,
            num_classes=3,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
            )
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            num_filters * len(kernel_sizes),
            num_classes,
            )

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)

        features = []

        for conv in self.convs:
            h = torch.relu(conv(x))
            pooled = torch.max(h, dim=2).values
            features.append(pooled)

        x = torch.cat(features, dim=1)
        x = self.dropout(x)

        return self.fc(x)


# 5. VALIDATION

def evaluate(model, loader, criterion):
    model.eval()

    all_true = []
    all_pred = []
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            predictions = torch.argmax(logits, dim=1)

            batch_size = len(y)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_true.extend(y.cpu().tolist())
            all_pred.extend(predictions.cpu().tolist())

    return {
        "val_loss": total_loss / total_samples,
        "macro_f1": f1_score(
            all_true, all_pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        ),
        "neutral_f1": f1_score(
            all_true, all_pred,
            labels=[1],
            average=None,
            zero_division=0,
        )[0],
        "accuracy": accuracy_score(all_true, all_pred),
        "y_true": all_true,
        "y_pred": all_pred,
    }



# 6. TRAIN MOT BO THAM SO

def train_config(
        config,
        train_loader,
        val_loader,
        vocab_size,
        class_weights,
        epochs,
        patience,
        seed,
):
    set_seed(seed)

    model = TextCNN(
        vocab_size=vocab_size,
        embedding_dim=config["embedding_dim"],
        num_filters=config["num_filters"],
        kernel_sizes=KERNEL_SIZES,
        dropout=config["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.01,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    best_score = -1.0
    best_epoch = 0
    best_state = None
    best_eval = None

    wait = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0
        total_samples = 0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            total_samples += len(y)

        train_loss = total_loss / total_samples
        result = evaluate(model, val_loader, criterion)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": result["val_loss"],
            "val_macro_f1": result["macro_f1"],
            "val_neutral_f1": result["neutral_f1"],
            "val_accuracy": result["accuracy"],
        })

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss={train_loss:.4f} | "
            f"Val Loss={result['val_loss']:.4f} | "
            f"Macro F1={result['macro_f1']:.4f} | "
            f"Neutral F1={result['neutral_f1']:.4f}",
            flush=True,
        )

        if result["macro_f1"] > best_score:
            best_score = result["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_eval = result
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print("Early Stopping activated.")
            break

    return {
        "best_score": float(best_score),
        "best_epoch": best_epoch,
        "best_state": best_state,
        "best_eval": best_eval,
        "history": history,
    }


# 7. TIM BO THAM SO TOT NHAT

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    torch.set_num_threads(4)

    train, val = load_data(smoke=args.smoke)

    train_texts = train["text_segmented"].fillna("")
    val_texts = val["text_segmented"].fillna("")

    vocab = build_vocab(train_texts)

    X_train = encode_texts(
        train_texts, vocab, args.max_length
    )
    X_val = encode_texts(
        val_texts, vocab, args.max_length
    )

    y_train = torch.tensor(
        train["sentiment"].to_numpy(),
        dtype=torch.long,
    )
    y_val = torch.tensor(
        val["sentiment"].to_numpy(),
        dtype=torch.long,
    )

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Xu ly mat can bang nhan
    counts = np.bincount(
        y_train.numpy(), minlength=3
    )

    weights = len(y_train) / (
            3 * np.maximum(counts, 1)
    )

    class_weights = torch.tensor(
        weights, dtype=torch.float32
    ).to(DEVICE)

    print("Vocabulary size:", len(vocab))
    print("Class weights:", class_weights.tolist())
    print("Device:", DEVICE)

    # Danh sach bo tham so can thu
    if args.smoke:
        configs = [
            (128, 64, 0.3, 0.001)
        ]
    elif args.full:
        configs = list(product(
            [128],
            [64, 100],
            [0.3, 0.5],
            [0.001, 0.0005],
        ))
    else:
        configs = list(product(
            [128],
            [64, 100],
            [0.3, 0.5],
            [0.001],
        ))

    max_epochs = 1 if args.smoke else args.epochs

    output_root = ROOT / "results"

    if args.smoke:
        output_root = output_root / "smoke"

    metrics_dir = output_root / "metrics"
    history_dir = output_root / "training_history"
    cm_dir = output_root / "confusion_matrix"

    checkpoint_dir = ROOT / "checkpoints"
    if args.smoke:
        checkpoint_dir = checkpoint_dir / "smoke"

    for directory in [
        metrics_dir, history_dir,
        cm_dir, checkpoint_dir
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    trials = []
    all_history = []

    global_best = -1.0
    global_params = None
    global_eval = None

    for trial_id, values in enumerate(configs, 1):
        embedding_dim, num_filters, dropout, lr = values

        config = {
            "embedding_dim": embedding_dim,
            "num_filters": num_filters,
            "dropout": dropout,
            "learning_rate": lr,
            "kernel_sizes": list(KERNEL_SIZES),
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "class_weight": "balanced",
        }

        print(
            f"\n===== TEXTCNN {trial_id}/{len(configs)} ====="
        )
        print("Parameters:", config, flush=True)

        result = train_config(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            vocab_size=len(vocab),
            class_weights=class_weights,
            epochs=max_epochs,
            patience=args.patience,
            seed=args.seed,
        )

        trials.append({
            "trial": trial_id,
            **config,
            "best_epoch": result["best_epoch"],
            "val_macro_f1": result["best_score"],
            "val_neutral_f1": float(
                result["best_eval"]["neutral_f1"]
            ),
            "val_accuracy": float(
                result["best_eval"]["accuracy"]
            ),
        })

        for record in result["history"]:
            all_history.append({
                "trial": trial_id,
                **config,
                **record,
            })

        print(
            f"BEST trial {trial_id}: "
            f"Macro F1={result['best_score']:.4f} | "
            f"Epoch={result['best_epoch']}"
        )

        if result["best_score"] > global_best:
            global_best = result["best_score"]
            global_params = config.copy()
            global_params["best_epoch"] = result["best_epoch"]
            global_eval = result["best_eval"]

            torch.save({
                "model_state_dict": result["best_state"],
                "vocab": vocab,
                "params": global_params,
                "seed": args.seed,
            }, checkpoint_dir / "textcnn_best.pt")

        # Luu ket qua sau moi trial
        pd.DataFrame(trials).to_csv(
            metrics_dir / "textcnn_trials.csv",
            index=False,
            encoding="utf-8-sig",
            )

        pd.DataFrame(all_history).to_csv(
            history_dir / "textcnn_history.csv",
            index=False,
            encoding="utf-8-sig",
            )

    # 8. LUU KET QUA TOT NHAT

    best_summary = {
        "model": "TextCNN",
        "selection_metric": "validation_macro_f1",
        "best_validation_macro_f1": global_best,
        "best_validation_neutral_f1": float(
            global_eval["neutral_f1"]
        ),
        "best_validation_accuracy": float(
            global_eval["accuracy"]
        ),
        "best_params": global_params,
        "seed": args.seed,
        "search_trials": len(configs),
    }

    with open(
            metrics_dir / "textcnn_best.json",
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(
            best_summary, f,
            ensure_ascii=False,
            indent=4,
        )

    cm = confusion_matrix(
        global_eval["y_true"],
        global_eval["y_pred"],
        labels=[0, 1, 2],
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=LABEL_NAMES,
    )

    disp.plot(cmap="Blues", colorbar=False)
    plt.title("TextCNN - Validation Confusion Matrix")
    plt.tight_layout()

    save_optimized_figure(plt.gcf(), 
        cm_dir / "textcnn_validation.webp",
        dpi=160,
        )
    plt.close()

    print("\n========== FINAL RESULT ==========")
    print("Best params:", global_params)
    print(f"Best Validation Macro F1: {global_best:.4f}")
    print(
        "Best Validation Neutral F1:",
        round(float(global_eval["neutral_f1"]), 4),
    )
    print("Saved results to:", output_root)
    print("Test dataset was NOT used.")


if __name__ == "__main__":
    main()
