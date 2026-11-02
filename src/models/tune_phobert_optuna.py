import argparse
import gc
import json
import math
import os
import random
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.optimization import Adafactor

from src.utils.image_saver import save_optimized_figure


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
RESULT_DIR = ROOT / "results" / "optuna" / "phobert"
CHECKPOINT_DIR = ROOT / "checkpoints" / "phobert_best"

MODEL_ID = "vinai/phobert-base-v2"
STUDY_NAME = "phobert_optuna_v1"
SEARCH_VERSION = "expanded_v2"

SEED = 42
LABELS = [0, 1, 2]
LABEL_NAMES = ["Negative", "Neutral", "Positive"]

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.tokens = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.tokens["input_ids"][index],
            "attention_mask": self.tokens["attention_mask"][index],
            "labels": self.labels[index],
        }


def load_split(filename):
    path = DATA_DIR / filename
    df = pd.read_csv(path)

    required = {"text_segmented", "sentiment"}

    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns in {filename}")

    texts = (
        df["text_segmented"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    labels = (
        df["sentiment"]
        .astype(int)
        .tolist()
    )

    if not set(labels).issubset(set(LABELS)):
        raise ValueError(f"Invalid labels in {filename}")

    return texts, labels


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def suggest_params(trial):
    params = {
        "learning_rate": trial.suggest_float(
            "learning_rate", 5e-6, 5e-5, log=True
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay", 0.0, 0.1
        ),
        "optimizer": trial.suggest_categorical(
            "optimizer", ["adamw", "adafactor"]
        ),
        "scheduler": trial.suggest_categorical(
            "scheduler", ["linear", "cosine", "constant"]
        ),
        "warmup_ratio": trial.suggest_float(
            "warmup_ratio", 0.0, 0.15
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size", [1, 2]
        ),
        "gradient_accumulation": trial.suggest_categorical(
            "gradient_accumulation", [4, 8, 16]
        ),
        "max_length": trial.suggest_categorical(
            "max_length", [64, 96, 128]
        ),
        "hidden_dropout_prob": trial.suggest_float(
            "hidden_dropout_prob", 0.05, 0.3
        ),
        "attention_probs_dropout_prob": trial.suggest_float(
            "attention_probs_dropout_prob", 0.0, 0.2
        ),
        "classifier_dropout": trial.suggest_float(
            "classifier_dropout", 0.05, 0.4
        ),
        "balanced_loss": trial.suggest_categorical(
            "balanced_loss", [False, True]
        ),
        "label_smoothing": trial.suggest_float(
            "label_smoothing", 0.0, 0.15
        ),
    }

    if params["optimizer"] == "adamw":
        params["max_grad_norm"] = trial.suggest_categorical(
            "max_grad_norm", [0.5, 1.0, 2.0]
        )

        params["adam_beta1"] = trial.suggest_float(
            "adam_beta1", 0.85, 0.95
        )

        params["adam_beta2"] = trial.suggest_categorical(
            "adam_beta2", [0.98, 0.999]
        )

        params["adam_epsilon"] = trial.suggest_categorical(
            "adam_epsilon", [1e-8, 1e-6]
        )

    else:
        params["adafactor_clip_threshold"] = (
            trial.suggest_categorical(
                "adafactor_clip_threshold", [1.0, 2.0]
            )
        )

    return params


def build_optimizer(model, params):
    decay_params = []
    no_decay_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        name_lower = name.lower()

        if (
                name.endswith("bias")
                or "layernorm" in name_lower
                or "layer_norm" in name_lower
        ):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    parameter_groups = [
        {
            "params": decay_params,
            "weight_decay": params["weight_decay"],
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]

    if params["optimizer"] == "adamw":
        return torch.optim.AdamW(
            parameter_groups,
            lr=params["learning_rate"],
            betas=(
                params["adam_beta1"],
                params["adam_beta2"],
            ),
            eps=params["adam_epsilon"],
        )

    return Adafactor(
        parameter_groups,
        lr=params["learning_rate"],
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        clip_threshold=params["adafactor_clip_threshold"],
    )


def build_scheduler(optimizer, params, total_steps):
    warmup_steps = int(
        total_steps * params["warmup_ratio"]
    )

    def lr_lambda(step):
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

        scheduler_type = params["scheduler"]

        if scheduler_type == "linear":
            return max(0.0, 1.0 - progress)

        if scheduler_type == "cosine":
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

    predictions = []
    targets = []

    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(
                device,
                non_blocking=True,
            )

            attention_mask = batch["attention_mask"].to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
            ):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            predicted = outputs.logits.argmax(dim=-1)

            predictions.extend(
                predicted.cpu().tolist()
            )

            targets.extend(
                batch["labels"].tolist()
            )

    return {
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=LABELS,
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
        "predictions": predictions,
    }


def save_best_checkpoint(
        model,
        tokenizer,
        trial,
        validation_labels,
        metrics,
):
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        CHECKPOINT_DIR,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(CHECKPOINT_DIR)

    report = classification_report(
        validation_labels,
        metrics["predictions"],
        labels=LABELS,
        target_names=LABEL_NAMES,
        zero_division=0,
    )

    (RESULT_DIR / "validation_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        validation_labels,
        metrics["predictions"],
        labels=LABELS,
        display_labels=LABEL_NAMES,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )

    ax.set_title("PhoBERT Validation Confusion Matrix")

    fig.tight_layout()

    save_optimized_figure(
        fig,
        RESULT_DIR / "confusion_matrix.webp",
        dpi=160,
        )

    plt.close(fig)

    checkpoint_info = {
        "trial": trial.number,
        "best_epoch": trial.user_attrs["best_epoch"],
        "validation_macro_f1": metrics["macro_f1"],
    }

    with open(
            RESULT_DIR / "best_checkpoint_info.json",
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(
            checkpoint_info,
            f,
            ensure_ascii=False,
            indent=4,
        )


def export_results(study, train_size, validation_size):
    study.trials_dataframe().to_csv(
        RESULT_DIR / "trials.csv",
        index=False,
        encoding="utf-8-sig",
        )

    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed:
        return

    best = study.best_trial

    summary = {
        "model": "phobert",
        "pretrained_model": MODEL_ID,
        "optimizer": "Optuna TPE",
        "selection_metric": "validation_macro_f1",
        "study_name": STUDY_NAME,
        "best_trial": best.number,
        "best_epoch": best.user_attrs.get("best_epoch"),
        "best_params": best.params,
        "validation_macro_f1": float(best.value),
        "validation_neutral_f1": best.user_attrs.get(
            "validation_neutral_f1"
        ),
        "validation_accuracy": best.user_attrs.get(
            "validation_accuracy"
        ),
        "completed_trials": len(completed),
        "total_trials": len(study.trials),
        "epochs_limit": study.user_attrs["settings"]["epochs"],
        "patience": study.user_attrs["settings"]["patience"],
        "seed": SEED,
        "train_rows": train_size,
        "validation_rows": validation_size,
        "checkpoint": str(
            CHECKPOINT_DIR.relative_to(ROOT)
        ),
        "test_used": False,
    }

    with open(
            RESULT_DIR / "best.json",
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=4,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trials",
        type=int,
        default=5,
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

    if min(args.trials, args.epochs, args.patience) < 1:
        parser.error("All numeric arguments must be >= 1")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. Check PyTorch installation."
        )

    set_seed(SEED)

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_texts, train_labels = load_split("train.csv")
    val_texts, val_labels = load_split("validation.csv")

    if set(train_labels) != set(LABELS):
        raise ValueError(
            "Train must contain labels 0, 1, 2"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=False,
    )

    device = torch.device("cuda")

    dataset_cache = {}

    print("Train:", len(train_texts), flush=True)
    print("Validation:", len(val_texts), flush=True)
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
        flush=True,
    )
    print("Test dataset is not used.", flush=True)

    database_path = RESULT_DIR / "study.db"

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{database_path.as_posix()}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=SEED,
            n_startup_trials=10,
        ),
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )

    settings = {
        "model": MODEL_ID,
        "search_version": SEARCH_VERSION,
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": SEED,
    }

    existing_settings = study.user_attrs.get("settings")

    if study.trials and existing_settings != settings:
        raise RuntimeError(
            "Existing study uses different settings. "
            "Back up and remove study.db before restarting."
        )

    study.set_user_attr("settings", settings)

    print(
        "Existing trials:",
        len(study.trials),
        flush=True,
    )

    print(
        "Additional trials:",
        args.trials,
        flush=True,
    )

    def get_datasets(max_length):
        if max_length not in dataset_cache:
            dataset_cache[max_length] = (
                SentimentDataset(
                    train_texts,
                    train_labels,
                    tokenizer,
                    max_length,
                ),
                SentimentDataset(
                    val_texts,
                    val_labels,
                    tokenizer,
                    max_length,
                ),
            )

        return dataset_cache[max_length]

    def objective(trial):
        set_seed(SEED + trial.number)

        params = suggest_params(trial)

        train_dataset, val_dataset = get_datasets(
            params["max_length"]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        model = None
        optimizer = None
        scheduler = None
        scaler = None

        try:
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

            config.hidden_dropout_prob = (
                params["hidden_dropout_prob"]
            )

            config.attention_probs_dropout_prob = (
                params["attention_probs_dropout_prob"]
            )

            config.classifier_dropout = (
                params["classifier_dropout"]
            )

            model = (
                AutoModelForSequenceClassification
                .from_pretrained(
                    MODEL_ID,
                    config=config,
                )
            )

            model.gradient_checkpointing_enable()
            model.to(device)

            optimizer = build_optimizer(
                model,
                params,
            )

            accumulation = params[
                "gradient_accumulation"
            ]

            steps_per_epoch = math.ceil(
                len(train_loader) / accumulation
            )

            total_steps = (
                    steps_per_epoch * args.epochs
            )

            scheduler = build_scheduler(
                optimizer,
                params,
                total_steps,
            )

            scaler = torch.amp.GradScaler("cuda")

            class_weights = None

            if params["balanced_loss"]:
                counts = np.bincount(
                    train_labels,
                    minlength=3,
                ).astype(np.float64)

                weights = (
                        len(train_labels)
                        / (3.0 * counts)
                )

                class_weights = torch.tensor(
                    weights,
                    dtype=torch.float32,
                    device=device,
                )

            best_f1 = -1.0
            best_epoch = 0
            best_metrics = None
            stale_epochs = 0
            history = []

            with tempfile.TemporaryDirectory(
                    prefix="phobert_trial_",
                    dir=RESULT_DIR,
            ) as temporary_directory:

                best_weights_path = (
                        Path(temporary_directory)
                        / "best_epoch.pt"
                )

                for epoch in range(
                        1,
                        args.epochs + 1,
                ):
                    model.train()

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    total_loss = 0.0

                    for index, batch in enumerate(
                            train_loader
                    ):
                        input_ids = batch[
                            "input_ids"
                        ].to(
                            device,
                            non_blocking=True,
                        )

                        attention_mask = batch[
                            "attention_mask"
                        ].to(
                            device,
                            non_blocking=True,
                        )

                        labels = batch[
                            "labels"
                        ].to(
                            device,
                            non_blocking=True,
                        )

                        with torch.autocast(
                                device_type="cuda",
                                dtype=torch.float16,
                        ):
                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
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

                        scaled_loss = (
                                loss / group_size
                        )

                        scaler.scale(
                            scaled_loss
                        ).backward()

                        total_loss += float(
                            loss.item()
                        )

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

                            previous_scale = (
                                scaler.get_scale()
                            )

                            scaler.step(optimizer)
                            scaler.update()

                            if (
                                    scaler.get_scale()
                                    >= previous_scale
                            ):
                                scheduler.step()

                            optimizer.zero_grad(
                                set_to_none=True
                            )

                        if (
                                (index + 1) % 2000 == 0
                                or index + 1 == len(train_loader)
                        ):
                            print(
                                f"Trial {trial.number} | "
                                f"Epoch {epoch}/{args.epochs} | "
                                f"Batch {index + 1}/"
                                f"{len(train_loader)}",
                                flush=True,
                            )

                    metrics = evaluate(
                        model,
                        val_loader,
                        device,
                    )

                    average_loss = (
                            total_loss / len(train_loader)
                    )

                    history.append({
                        "epoch": epoch,
                        "train_loss": average_loss,
                        "validation_macro_f1": (
                            metrics["macro_f1"]
                        ),
                        "validation_neutral_f1": (
                            metrics["neutral_f1"]
                        ),
                        "validation_accuracy": (
                            metrics["accuracy"]
                        ),
                    })

                    print(
                        f"Trial {trial.number} | "
                        f"Epoch {epoch}/{args.epochs} | "
                        f"Loss {average_loss:.4f} | "
                        f"Macro F1 {metrics['macro_f1']:.4f} | "
                        f"Neutral F1 {metrics['neutral_f1']:.4f}",
                        flush=True,
                    )

                    if metrics["macro_f1"] > best_f1:
                        best_f1 = metrics["macro_f1"]
                        best_epoch = epoch
                        best_metrics = metrics

                        torch.save(
                            model.state_dict(),
                            best_weights_path,
                        )

                        stale_epochs = 0
                    else:
                        stale_epochs += 1

                    trial.set_user_attr(
                        "history",
                        history,
                    )

                    if stale_epochs >= args.patience:
                        print(
                            f"Trial {trial.number}: "
                            "Early stopping",
                            flush=True,
                        )
                        break

                trial.set_user_attr(
                    "best_epoch",
                    best_epoch,
                )

                trial.set_user_attr(
                    "validation_neutral_f1",
                    best_metrics["neutral_f1"],
                )

                trial.set_user_attr(
                    "validation_accuracy",
                    best_metrics["accuracy"],
                )

                completed_trials = [
                    old_trial
                    for old_trial in study.trials
                    if old_trial.state
                       == optuna.trial.TrialState.COMPLETE
                ]

                previous_best = max(
                    (
                        old_trial.value
                        for old_trial in completed_trials
                    ),
                    default=-1.0,
                )

                if best_f1 > previous_best:
                    state_dict = torch.load(
                        best_weights_path,
                        map_location="cpu",
                        weights_only=True,
                    )

                    model.load_state_dict(
                        state_dict
                    )

                    del state_dict

                    save_best_checkpoint(
                        model,
                        tokenizer,
                        trial,
                        val_labels,
                        best_metrics,
                    )

                    print(
                        f"New best checkpoint: "
                        f"Trial {trial.number} | "
                        f"Macro F1 {best_f1:.4f}",
                        flush=True,
                    )

                return best_f1

        except torch.cuda.OutOfMemoryError:
            print(
                f"Trial {trial.number}: "
                "GPU out of memory. Trial pruned.",
                flush=True,
            )

            raise optuna.TrialPruned(
                "GPU out of memory"
            )

        finally:
            del model
            del optimizer
            del scheduler
            del scaler

            gc.collect()
            torch.cuda.empty_cache()

    def callback(current_study, trial):
        export_results(
            current_study,
            len(train_texts),
            len(val_texts),
        )

    try:
        study.optimize(
            objective,
            n_trials=args.trials,
            n_jobs=1,
            callbacks=[callback],
            gc_after_trial=True,
        )

    except KeyboardInterrupt:
        print(
            "Training interrupted. "
            "Completed trials remain saved.",
            flush=True,
        )

    finally:
        export_results(
            study,
            len(train_texts),
            len(val_texts),
        )

        print(
            "Results:",
            RESULT_DIR,
            flush=True,
        )

        print(
            "Checkpoint:",
            CHECKPOINT_DIR,
            flush=True,
        )


if __name__ == "__main__":
    main()