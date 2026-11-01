import argparse
import gc
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.optimization import Adafactor


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
STUDY_DIR = ROOT / "results" / "optuna" / "phobert"
RESUME_DIR = ROOT / "checkpoints" / "phobert_resume"

MODEL_ID = "vinai/phobert-base-v2"
STUDY_NAME = "phobert_optuna_v1"
SEED = 42

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_data(filename):
    df = pd.read_csv(DATA_DIR / filename)

    texts = (
        df["text_segmented"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    labels = df["sentiment"].astype(int).tolist()

    return texts, labels


def save_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )


def save_atomic(data, destination):
    temporary = destination.with_name(
        destination.name + ".tmp"
    )

    torch.save(data, temporary)
    os.replace(temporary, destination)


def prepare(trial_number, best_epoch, epochs):
    RESUME_DIR.mkdir(parents=True, exist_ok=True)

    weights_path = RESUME_DIR / "best_epoch.pt"

    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    db_path = STUDY_DIR / "study.db"

    if not db_path.exists():
        raise FileNotFoundError(db_path)

    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{db_path.as_posix()}",
    )

    selected = [
        trial
        for trial in study.trials
        if trial.number == trial_number
    ]

    if not selected:
        raise ValueError("Trial not found")

    trial = selected[0]

    if not trial.params:
        raise ValueError("Trial has no parameters")

    metadata = {
        "model": MODEL_ID,
        "trial_number": trial.number,
        "params": trial.params,
        "best_epoch": best_epoch,
        "target_epochs": epochs,
        "seed": SEED,
    }

    save_json(
        RESUME_DIR / "resume_info.json",
        metadata,
        )

    print("Resume metadata saved.")
    print("Trial:", trial.number)
    print("Checkpoint epoch:", best_epoch)
    print("Target epochs:", epochs)
    print("Weights:", weights_path)


def build_optimizer(model, params):
    decay = []
    no_decay = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        name_lower = name.lower()

        if (
                name.endswith("bias")
                or "layernorm" in name_lower
                or "layer_norm" in name_lower
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    groups = [
        {
            "params": decay,
            "weight_decay": params["weight_decay"],
        },
        {
            "params": no_decay,
            "weight_decay": 0.0,
        },
    ]

    if params["optimizer"] == "adamw":
        return torch.optim.AdamW(
            groups,
            lr=params["learning_rate"],
            betas=(
                params["adam_beta1"],
                params["adam_beta2"],
            ),
            eps=params["adam_epsilon"],
        )

    return Adafactor(
        groups,
        lr=params["learning_rate"],
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        clip_threshold=params["adafactor_clip_threshold"],
    )


def build_scheduler(
        optimizer,
        params,
        total_steps,
        offset_steps,
):
    warmup_steps = int(
        total_steps * params["warmup_ratio"]
    )

    def lr_lambda(current_step):
        step = current_step + offset_steps

        if step < warmup_steps:
            return max(
                1e-8,
                (step + 1) / max(1, warmup_steps),
                )

        progress = min(
            1.0,
            (step - warmup_steps)
            / max(1, total_steps - warmup_steps),
            )

        if params["scheduler"] == "linear":
            return max(0.0, 1.0 - progress)

        if params["scheduler"] == "cosine":
            return max(
                0.0,
                0.5 * (1.0 + math.cos(math.pi * progress)),
                )

        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )


def evaluate(model, loader, device):
    model.eval()

    targets = []
    predictions = []

    with torch.inference_mode():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
            ):
                outputs = model(
                    input_ids=ids,
                    attention_mask=mask,
                )

            predictions.extend(
                outputs.logits.argmax(dim=-1).cpu().tolist()
            )

            targets.extend(batch["labels"].tolist())

    return {
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
        "neutral_f1": float(
            f1_score(
                targets,
                predictions,
                labels=[1],
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(
            accuracy_score(targets, predictions)
        ),
    }


def resume_training(patience):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    metadata_path = RESUME_DIR / "resume_info.json"

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    params = metadata["params"]
    target_epochs = metadata["target_epochs"]

    set_seed(metadata["seed"])

    device = torch.device("cuda")

    train_texts, train_labels = read_data("train.csv")
    val_texts, val_labels = read_data("validation.csv")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=False,
    )

    train_dataset = SentimentDataset(
        train_texts,
        train_labels,
        tokenizer,
        params["max_length"],
    )

    val_dataset = SentimentDataset(
        val_texts,
        val_labels,
        tokenizer,
        params["max_length"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    config = AutoConfig.from_pretrained(
        MODEL_ID,
        num_labels=3,
        id2label={
            0: "Negative",
            1: "Neutral",
            2: "Positive",
        },
        label2id={
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2,
        },
    )

    config.hidden_dropout_prob = params[
        "hidden_dropout_prob"
    ]

    config.attention_probs_dropout_prob = params[
        "attention_probs_dropout_prob"
    ]

    config.classifier_dropout = params[
        "classifier_dropout"
    ]

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_ID,
            config=config,
        )
    )

    model.gradient_checkpointing_enable()

    checkpoint_path = RESUME_DIR / "training_state.pt"
    best_weights_path = RESUME_DIR / "best_epoch.pt"

    if checkpoint_path.exists():
        saved = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        model.load_state_dict(saved["model"])

        start_epoch = saved["epoch"] + 1
        best_epoch = saved["best_epoch"]
        best_f1 = saved["best_f1"]
        stale_epochs = saved["stale_epochs"]
        history = saved["history"]

        print(
            "Full checkpoint found.",
            "Continue from epoch:",
            start_epoch,
            flush=True,
        )

    else:
        weights = torch.load(
            best_weights_path,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(weights)
        del weights

        saved = None
        best_epoch = metadata["best_epoch"]
        start_epoch = best_epoch + 1
        stale_epochs = 0
        history = []

        print(
            "Recovered model weights.",
            "Continue from epoch:",
            start_epoch,
            flush=True,
        )

    model.to(device)

    optimizer = build_optimizer(model, params)

    accumulation = params["gradient_accumulation"]

    batches_per_epoch = math.ceil(
        len(train_dataset) / params["batch_size"]
    )

    steps_per_epoch = math.ceil(
        batches_per_epoch / accumulation
    )

    scheduler = build_scheduler(
        optimizer,
        params,
        total_steps=steps_per_epoch * target_epochs,
        offset_steps=(
                metadata["best_epoch"] * steps_per_epoch
        ),
    )

    scaler = torch.amp.GradScaler("cuda")

    if saved is not None:
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        del saved

    else:
        initial_metrics = evaluate(
            model,
            val_loader,
            device,
        )

        best_f1 = initial_metrics["macro_f1"]

        print(
            f"Recovered Validation Macro F1: "
            f"{best_f1:.4f}",
            flush=True,
        )

    class_weights = None

    if params["balanced_loss"]:
        counts = np.bincount(
            train_labels,
            minlength=3,
        ).astype(np.float64)

        values = len(train_labels) / (3.0 * counts)

        class_weights = torch.tensor(
            values,
            dtype=torch.float32,
            device=device,
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Batch size:", params["batch_size"])
    print("Accumulation:", accumulation)
    print("Target epochs:", target_epochs)
    print("Test is not used.")

    for epoch in range(start_epoch, target_epochs + 1):
        set_seed(SEED + epoch)

        train_loader = DataLoader(
            train_dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0

        for index, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(
                device,
                non_blocking=True,
            )

            mask = batch["attention_mask"].to(
                device,
                non_blocking=True,
            )

            labels = batch["labels"].to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
            ):
                outputs = model(
                    input_ids=ids,
                    attention_mask=mask,
                )

                loss = F.cross_entropy(
                    outputs.logits.float(),
                    labels,
                    weight=class_weights,
                    label_smoothing=params[
                        "label_smoothing"
                    ],
                )

            group_start = (
                                  index // accumulation
                          ) * accumulation

            group_size = min(
                accumulation,
                len(train_loader) - group_start,
                )

            scaler.scale(
                loss / group_size
            ).backward()

            total_loss += loss.item()

            should_step = (
                    (index + 1) % accumulation == 0
                    or index + 1 == len(train_loader)
            )

            if should_step:
                if params["optimizer"] == "adamw":
                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        params["max_grad_norm"],
                    )

                previous_scale = scaler.get_scale()

                scaler.step(optimizer)
                scaler.update()

                if scaler.get_scale() >= previous_scale:
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)

            if (
                    (index + 1) % 2000 == 0
                    or index + 1 == len(train_loader)
            ):
                print(
                    f"Epoch {epoch}/{target_epochs} | "
                    f"Batch {index + 1}/{len(train_loader)}",
                    flush=True,
                )

        metrics = evaluate(
            model,
            val_loader,
            device,
        )

        average_loss = total_loss / len(train_loader)

        history.append({
            "epoch": epoch,
            "train_loss": average_loss,
            **metrics,
        })

        print(
            f"Epoch {epoch}/{target_epochs} | "
            f"Loss {average_loss:.4f} | "
            f"Macro F1 {metrics['macro_f1']:.4f} | "
            f"Neutral F1 {metrics['neutral_f1']:.4f}",
            flush=True,
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0

            save_atomic(
                model.state_dict(),
                best_weights_path,
            )

            print(
                f"New best epoch: {best_epoch} | "
                f"Macro F1: {best_f1:.4f}",
                flush=True,
            )

        else:
            stale_epochs += 1

        state = {
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_f1": best_f1,
            "stale_epochs": stale_epochs,
            "history": history,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        }

        save_atomic(
            state,
            checkpoint_path,
        )

        save_json(
            RESUME_DIR / "training_progress.json",
            {
                "last_epoch": epoch,
                "best_epoch": best_epoch,
                "best_macro_f1": best_f1,
                "target_epochs": target_epochs,
                "completed": epoch == target_epochs,
            },
            )

        print(
            "Checkpoint saved after epoch",
            epoch,
            flush=True,
        )

        del state
        gc.collect()
        torch.cuda.empty_cache()

        if stale_epochs >= patience:
            print("Early stopping.", flush=True)
            break

    print("Training finished or stopped.")
    print("Best epoch:", best_epoch)
    print("Best Macro F1:", best_f1)
    print("Checkpoint:", checkpoint_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prepare",
        action="store_true",
    )

    parser.add_argument(
        "--trial",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--best-epoch",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    if args.prepare:
        prepare(
            args.trial,
            args.best_epoch,
            args.epochs,
        )
    else:
        resume_training(args.patience)


if __name__ == "__main__":
    main()