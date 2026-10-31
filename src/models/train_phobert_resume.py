
import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "vinai/phobert-base-v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Trial 1 trung voi cau hinh PhoBERT tot nhat da tim duoc.
CONFIGS = {
    1: {"learning_rate": 2e-5, "batch_size": 1, "accum": 8},
    2: {"learning_rate": 3e-5, "batch_size": 1, "accum": 8},
    3: {"learning_rate": 2e-5, "batch_size": 2, "accum": 4},
    4: {"learning_rate": 3e-5, "batch_size": 2, "accum": 4},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(path, tokenizer):
    df = pd.read_csv(path)
    texts = df["text_segmented"].fillna("").astype(str).tolist()
    labels = df["sentiment"].astype(int).to_numpy()

    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_attention_mask=True,
    )

    dataset = TensorDataset(
        torch.tensor(encoded["input_ids"], dtype=torch.long),
        torch.tensor(encoded["attention_mask"], dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )
    return dataset, labels


def evaluate(model, loader, criterion):
    model.eval()
    predictions, targets = [], []
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for ids, mask, labels in loader:
            ids = ids.to(DEVICE)
            mask = mask.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(
                input_ids=ids,
                attention_mask=mask,
            ).logits

            loss = criterion(logits, labels)
            pred = logits.argmax(dim=1)

            total_loss += loss.item() * len(labels)
            total_items += len(labels)
            predictions.extend(pred.cpu().tolist())
            targets.extend(labels.cpu().tolist())

    f1 = f1_score(
        targets, predictions,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    return {
        "val_loss": total_loss / total_items,
        "macro_f1": float(np.mean(f1)),
        "neutral_f1": float(f1[1]),
        "accuracy": float(accuracy_score(targets, predictions)),
    }


def save_training_state(
        path, model, optimizer, scheduler,
        epoch, best_epoch, best_f1, history,
        config, total_epochs,
):
    # Luu vao file tam, sau do thay file cu.
    # Neu mat dien khi dang luu, checkpoint cu van con.
    temporary = path.with_suffix(".tmp")

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_f1": best_f1,
        "history": history,
        "config": config,
        "total_epochs": total_epochs,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None
        ),
    }

    torch.save(state, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--warm-start-temp",
        action="store_true",
        help="Use old temporary model weights, but reset optimizer.",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be >= 1")

    config = CONFIGS[args.trial].copy()
    config["max_length"] = 64
    config["weight_decay"] = 0.01

    set_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    print("Device:", DEVICE, flush=True)
    print("Trial:", args.trial, flush=True)
    print("Config:", config, flush=True)

    checkpoint_dir = ROOT / "checkpoints" / "phobert_resume"
    result_dir = ROOT / "results" / "phobert_resume"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    last_path = checkpoint_dir / f"trial_{args.trial}_last.pt"
    best_dir = checkpoint_dir / f"trial_{args.trial}_best"
    history_path = result_dir / f"trial_{args.trial}_history.csv"
    summary_path = result_dir / f"trial_{args.trial}_best.json"

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
    )

    train_ds, train_labels = make_dataset(
        ROOT / "data" / "processed" / "train.csv",
        tokenizer,
        )
    val_ds, _ = make_dataset(
        ROOT / "data" / "processed" / "validation.csv",
        tokenizer,
        )

    print("Train:", len(train_ds))
    print("Validation:", len(val_ds))

    counts = np.bincount(train_labels, minlength=3)
    class_weights = len(train_labels) / (3 * np.maximum(counts, 1))

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label={0: "Negative", 1: "Neutral", 2: "Positive"},
        label2id={"Negative": 0, "Neutral": 1, "Positive": 2},
    ).to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    batches_per_epoch = math.ceil(len(train_ds) / config["batch_size"])
    steps_per_epoch = math.ceil(batches_per_epoch / config["accum"])
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * 0.1))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(
            0.0,
            (total_steps - step) /
            max(1, total_steps - warmup_steps),
            )

    scheduler = LambdaLR(optimizer, lr_lambda)

    start_epoch = 1
    best_epoch = 0
    best_f1 = -1.0
    history = []

    if last_path.exists():
        print("Loading resume checkpoint:", last_path, flush=True)

        # Chi nap checkpoint do chinh script nay tao.
        state = torch.load(
            last_path,
            map_location="cpu",
            weights_only=False,
        )

        if state["config"] != config:
            raise RuntimeError("Checkpoint config does not match.")

        if state["total_epochs"] != args.epochs:
            raise RuntimeError("Checkpoint total epochs does not match.")

        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

        start_epoch = state["epoch"] + 1
        best_epoch = state["best_epoch"]
        best_f1 = state["best_f1"]
        history = state["history"]

        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])

        if torch.cuda.is_available() and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])

        del state

        print(f"RESUMED FROM EPOCH {start_epoch}", flush=True)

    elif args.warm_start_temp:
        old_path = ROOT / "checkpoints" / "_phobert_trial_best.pt"

        if not old_path.exists():
            raise FileNotFoundError(old_path)

        weights = torch.load(
            old_path,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(weights)
        del weights

        print(
            "WARM START: loaded old weights, "
            "but optimizer and epoch counter start fresh.",
            flush=True,
        )

    if start_epoch > args.epochs:
        print("This trial already completed all epochs.")
        return

    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        # Thu tu shuffle co dinh cho moi epoch.
        generator = torch.Generator()
        generator.manual_seed(args.seed + epoch)

        train_loader = DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            generator=generator,
            num_workers=0,
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        total_items = 0

        for batch_idx, (ids, mask, labels) in enumerate(train_loader, 1):
            ids = ids.to(DEVICE)
            mask = mask.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(
                input_ids=ids,
                attention_mask=mask,
            ).logits

            raw_loss = criterion(logits, labels)
            loss = raw_loss / config["accum"]
            loss.backward()

            running_loss += raw_loss.detach().item() * len(labels)
            total_items += len(labels)

            if (
                    batch_idx % config["accum"] == 0
                    or batch_idx == len(train_loader)
            ):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if batch_idx % 500 == 0:
                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss={running_loss / total_items:.4f}",
                    flush=True,
                )

        train_loss = running_loss / total_items
        metrics = evaluate(model, val_loader, criterion)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{args.epochs} DONE | "
            f"Macro F1={metrics['macro_f1']:.4f} | "
            f"Neutral F1={metrics['neutral_f1']:.4f}",
            flush=True,
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch

            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir, safe_serialization=True)
            tokenizer.save_pretrained(best_dir)

            summary_path.write_text(
                json.dumps(
                    {
                        "trial": args.trial,
                        "best_epoch": best_epoch,
                        "best_macro_f1": best_f1,
                        "config": config,
                        "seed": args.seed,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            print("Saved best model:", best_dir, flush=True)

        pd.DataFrame(history).to_csv(
            history_path,
            index=False,
            encoding="utf-8-sig",
        )

        save_training_state(
            last_path, model, optimizer, scheduler,
            epoch, best_epoch, best_f1,
            history, config, args.epochs,
        )

        print(
            f"CHECKPOINT SAVED | Epoch {epoch}/{args.epochs} | "
            f"Best F1={best_f1:.4f} at epoch {best_epoch}",
            flush=True,
        )

    print("TRAINING COMPLETED")
    print("Best epoch:", best_epoch)
    print("Best Macro F1:", best_f1)
    print("Checkpoint:", last_path)


if __name__ == "__main__":
    main()
