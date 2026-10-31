import argparse
import gc
import json
import math
import random
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
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# 1. GLOBAL CONFIGURATION

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"

MODEL_NAME = "vinai/phobert-base-v2"

LABEL_NAMES = [
    "Negative",
    "Neutral",
    "Positive",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 2. LOAD DATA

def stratified_smoke_sample(df, n, seed=42):
    """
    Lay mau nho, dam bao co du 3 nhan.
    Chi su dung cho Smoke Test.
    """
    rng = np.random.default_rng(seed)

    required_indices = []

    for label in [0, 1, 2]:
        candidates = df.index[
            df["sentiment"] == label
            ].to_numpy()

        if len(candidates) == 0:
            raise ValueError(
                f"Missing sentiment class: {label}"
            )

        required_indices.append(
            int(rng.choice(candidates))
        )

    remaining = df.drop(
        index=required_indices
    )

    extra_count = min(
        max(n - len(required_indices), 0),
        len(remaining),
    )

    if extra_count > 0:
        extra_indices = remaining.sample(
            n=extra_count,
            random_state=seed,
        ).index.tolist()
    else:
        extra_indices = []

    indices = (
            required_indices + extra_indices
    )

    return (
        df.loc[indices]
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def load_data(smoke=False, seed=42):
    train_path = (
            DATA_DIR / "train.csv"
    )

    val_path = (
            DATA_DIR / "validation.csv"
    )

    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing file: {train_path}"
        )

    if not val_path.exists():
        raise FileNotFoundError(
            f"Missing file: {val_path}"
        )

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    required_columns = {
        "text_segmented",
        "sentiment",
    }

    for name, df in [
        ("Train", train),
        ("Validation", val),
    ]:
        if not required_columns.issubset(
                df.columns
        ):
            raise ValueError(
                f"{name} missing required columns"
            )

        if df["sentiment"].isna().any():
            raise ValueError(
                f"{name} contains missing labels"
            )

        labels = set(
            df["sentiment"].unique()
        )

        if not labels.issubset(
                {0, 1, 2}
        ):
            raise ValueError(
                f"{name} has invalid labels: {labels}"
            )

    train["text_segmented"] = (
        train["text_segmented"]
        .fillna("")
        .astype(str)
    )

    val["text_segmented"] = (
        val["text_segmented"]
        .fillna("")
        .astype(str)
    )

    if smoke:
        train = stratified_smoke_sample(
            train, 12, seed
        )

        val = stratified_smoke_sample(
            val, 9, seed
        )

    print("Train:", len(train))
    print("Validation:", len(val))

    return train, val

# 3. TOKENIZATION

def create_dataset(
        df,
        tokenizer,
        max_length,
):
    texts = (
        df["text_segmented"]
        .tolist()
    )

    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )

    input_ids = torch.tensor(
        encoded["input_ids"],
        dtype=torch.long,
    )

    attention_masks = torch.tensor(
        encoded["attention_mask"],
        dtype=torch.long,
    )

    labels = torch.tensor(
        df["sentiment"].to_numpy(),
        dtype=torch.long,
    )

    return TensorDataset(
        input_ids,
        attention_masks,
        labels,
    )

# 4. VALIDATION EVALUATION

def evaluate(
        model,
        loader,
        criterion,
):
    model.eval()

    all_true = []
    all_pred = []

    total_loss = 0.0
    total_count = 0

    use_amp = (
            DEVICE.type == "cuda"
    )

    with torch.inference_mode():

        for (
                input_ids,
                attention_mask,
                labels,
        ) in loader:

            input_ids = input_ids.to(
                DEVICE
            )

            attention_mask = (
                attention_mask.to(DEVICE)
            )

            labels = labels.to(
                DEVICE
            )

            with torch.autocast(
                    device_type=DEVICE.type,
                    dtype=torch.float16,
                    enabled=use_amp,
            ):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            logits = (
                outputs.logits.float()
            )

            loss = criterion(
                logits,
                labels,
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            batch_count = len(labels)

            total_loss += (
                    loss.item() * batch_count
            )

            total_count += batch_count

            all_true.extend(
                labels.cpu().tolist()
            )

            all_pred.extend(
                predictions.cpu().tolist()
            )

    class_f1 = f1_score(
        all_true,
        all_pred,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    return {
        "val_loss": (
                total_loss / total_count
        ),
        "macro_f1": float(
            np.mean(class_f1)
        ),
        "neutral_f1": float(
            class_f1[1]
        ),
        "accuracy": float(
            accuracy_score(
                all_true,
                all_pred,
            )
        ),
        "y_true": all_true,
        "y_pred": all_pred,
    }

# 5. LEARNING RATE SCHEDULER

def create_scheduler(
        optimizer,
        total_steps,
        warmup_ratio=0.1,
):
    warmup_steps = int(
        total_steps * warmup_ratio
    )

    def lr_lambda(step):

        if step < warmup_steps:
            return float(
                step + 1
            ) / max(
                1,
                warmup_steps,
            )

        remaining_steps = (
                total_steps - step
        )

        decay_steps = max(
            1,
            total_steps - warmup_steps,
            )

        return max(
            0.0,
            float(remaining_steps)
            / decay_steps,
            )

    return LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )

# 6. TRAIN ONE CONFIGURATION

def train_one(
        config,
        train_dataset,
        val_dataset,
        class_weights,
        epochs,
        patience,
        seed,
        checkpoint_dir,
        tokenizer,
        global_best,
):
    set_seed(seed)

    use_amp = (
            DEVICE.type == "cuda"
    )

    batch_size = config[
        "batch_size"
    ]

    accumulation_steps = config.get(
        "gradient_accumulation_steps",
        1,
    )

    if accumulation_steps < 1:
        raise ValueError(
            "gradient_accumulation_steps must be >= 1"
        )

    print(
        "\nLoading pretrained PhoBERT...",
        flush=True,
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
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
    )

    # Giam GPU memory trong qua trinh train
    if use_amp:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False
            }
        )

    model = model.to(DEVICE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        foreach=False,
    )

    optimizer_steps_per_epoch = (
        math.ceil(
            len(train_loader)
            / accumulation_steps
        )
    )

    total_steps = (
            optimizer_steps_per_epoch
            * epochs
    )

    scheduler = create_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=0.1,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE)
    )

    best_score = -1.0
    best_epoch = 0
    best_eval = None

    history = []
    wait = 0

    temporary_checkpoint = (
            checkpoint_dir
            / "_phobert_trial_best.pt"
    )

    print(
        "Training device:", DEVICE,
        flush=True,
    )

    print(
        "Mixed Precision FP16:",
        use_amp,
        flush=True,
    )

    print(
        "Gradient Checkpointing:",
        use_amp,
        flush=True,
    )

    print(
        "Micro Batch Size:",
        batch_size,
        flush=True,
    )

    print(
        "Gradient Accumulation:",
        accumulation_steps,
        flush=True,
    )

    print(
        "Effective Batch Size:",
        batch_size * accumulation_steps,
        flush=True,
        )

    if use_amp:
        torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------
    # EPOCH LOOP
    # ------------------------------------------

    for epoch in range(
            1,
            epochs + 1,
    ):
        model.train()

        running_loss = 0.0
        total_count = 0

        optimizer.zero_grad(
            set_to_none=True
        )

        num_batches = len(
            train_loader
        )

        for batch_idx, (
                input_ids,
                attention_mask,
                labels,
        ) in enumerate(
            train_loader,
            start=1,
        ):

            input_ids = input_ids.to(
                DEVICE
            )

            attention_mask = (
                attention_mask.to(DEVICE)
            )

            labels = labels.to(
                DEVICE
            )

            # So batch trong nhom tich luy hien tai
            group_start = (
                                  (batch_idx - 1)
                                  // accumulation_steps
                          ) * accumulation_steps

            group_size = min(
                accumulation_steps,
                num_batches - group_start,
                )

            # ----------------------------------
            # FORWARD - FP16
            # ----------------------------------

            with torch.autocast(
                    device_type=DEVICE.type,
                    dtype=torch.float16,
                    enabled=use_amp,
            ):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                per_sample_loss = torch.nn.functional.cross_entropy(
                    outputs.logits.float(),
                    labels,
                    reduction="none",
                )

            raw_loss = (
                per_sample_loss * criterion.weight[labels]
            ).mean()

            loss = (
                    raw_loss / group_size
                )

            # ----------------------------------
            # BACKWARD
            # ----------------------------------

            scaler.scale(
                loss
            ).backward()

            running_loss += (
                    raw_loss.detach().item()
                    * len(labels)
            )

            total_count += len(
                labels
            )

            # ----------------------------------
            # OPTIMIZER UPDATE
            # ----------------------------------

            should_step = (
                    batch_idx % accumulation_steps == 0
                    or batch_idx == num_batches
            )

            if should_step:
                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                scale_before = (
                    scaler.get_scale()
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                scale_after = (
                    scaler.get_scale()
                )

                # Khong update scheduler neu
                # GradScaler bo qua optimizer step
                if scale_after >= scale_before:
                    scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

            # ----------------------------------
            # PRINT BATCH PROGRESS
            # ----------------------------------

            if (
                    batch_idx == 1
                    or batch_idx % 100 == 0
                    or batch_idx == num_batches
            ):
                current_loss = (
                        running_loss / total_count
                )

                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"Batch {batch_idx}/{num_batches} | "
                    f"Train Loss={current_loss:.4f}",
                    flush=True,
                )

        # --------------------------------------
        # VALIDATION AFTER EACH EPOCH
        # --------------------------------------

        train_loss = (
                running_loss / total_count
        )

        metrics = evaluate(
            model,
            val_loader,
            criterion,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics[
                "val_loss"
            ],
            "val_macro_f1": metrics[
                "macro_f1"
            ],
            "val_neutral_f1": metrics[
                "neutral_f1"
            ],
            "val_accuracy": metrics[
                "accuracy"
            ],
        })

        print(
            f"\nEpoch {epoch:02d}/{epochs} | "
            f"Train Loss={train_loss:.4f} | "
            f"Val Loss={metrics['val_loss']:.4f} | "
            f"Macro F1={metrics['macro_f1']:.4f} | "
            f"Neutral F1={metrics['neutral_f1']:.4f}",
            flush=True,
        )

        # --------------------------------------
        # SAVE BEST EPOCH
        # --------------------------------------

        if (
                metrics["macro_f1"]
                > best_score
        ):
            best_score = metrics[
                "macro_f1"
            ]

            best_epoch = epoch
            best_eval = metrics

            wait = 0

            # Luu checkpoint tam tren disk
            # thay vi copy model vao GPU RAM
            torch.save(
                model.state_dict(),
                temporary_checkpoint,
            )

            print(
                "New best checkpoint saved.",
                flush=True,
            )

        else:
            wait += 1

        # --------------------------------------
        # EARLY STOPPING
        # --------------------------------------

        if wait >= patience:
            print(
                "Early Stopping activated.",
                flush=True,
            )
            break

    # ------------------------------------------
    # SAVE GLOBAL BEST MODEL
    # ------------------------------------------

    if best_score > global_best:

        state_dict = torch.load(
            temporary_checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(
            state_dict
        )

        del state_dict

        best_model_dir = (
                checkpoint_dir
                / "phobert_best"
        )

        best_model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model.save_pretrained(
            best_model_dir,
            safe_serialization=True,
        )

        tokenizer.save_pretrained(
            best_model_dir
        )

        print(
            "Saved new best model:",
            best_model_dir,
            flush=True,
        )

    temporary_checkpoint.unlink(
        missing_ok=True
    )

    if use_amp:
        peak_memory_gb = (
                torch.cuda.max_memory_allocated()
                / 1024 ** 3
        )

        print(
            f"Peak allocated GPU memory: "
            f"{peak_memory_gb:.2f} GB",
            flush=True,
        )

    del model
    del optimizer
    del scheduler
    del scaler
    del criterion

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "best_score": best_score,
        "best_epoch": best_epoch,
        "best_eval": best_eval,
        "history": history,
    }


# ==================================================
# 7. SAVE RESULTS AND CHARTS
# ==================================================

def save_confusion_matrix(
        metrics,
        output_path,
):
    cm = confusion_matrix(
        metrics["y_true"],
        metrics["y_pred"],
        labels=[0, 1, 2],
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=LABEL_NAMES,
    )

    disp.plot(
        cmap="Blues",
        colorbar=False,
    )

    plt.title(
        "PhoBERT - Validation Confusion Matrix"
    )

    plt.tight_layout()

    save_optimized_figure(plt.gcf(), 
        output_path,
        dpi=160,
    )

    plt.close()


def save_history_chart(
        history,
        output_path,
):
    epochs = [
        item["epoch"]
        for item in history
    ]

    f1_values = [
        item["val_macro_f1"]
        for item in history
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        f1_values,
        marker="o",
        label="Validation Macro F1",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Macro F1"
    )

    plt.title(
        "PhoBERT - Validation Macro F1"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()
    plt.tight_layout()

    save_optimized_figure(plt.gcf(), 
        output_path,
        dpi=160,
    )

    plt.close()


# ==================================================
# 8. HYPERPARAMETER TUNING
# ==================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    parser.add_argument(
        "--full",
        action="store_true",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error(
            "--epochs must be >= 1"
        )

    if args.patience < 1:
        parser.error(
            "--patience must be >= 1"
        )

    if (
            args.trials is not None
            and args.trials < 1
    ):
        parser.error(
            "--trials must be >= 1"
        )

    set_seed(
        args.seed
    )

    torch.set_num_threads(
        min(
            4,
            torch.get_num_threads(),
        )
    )

    print(
        "Device:", DEVICE
    )

    print(
        "Pretrained model:", MODEL_NAME
    )

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        total_vram = (
                torch.cuda.get_device_properties(0)
                .total_memory
                / 1024 ** 3
        )

        print(
            f"Total GPU VRAM: "
            f"{total_vram:.2f} GB"
        )

    else:
        print(
            "WARNING: Running on CPU.",
            flush=True,
        )

    # ------------------------------------------
    # DATA
    # ------------------------------------------

    train, val = load_data(
        smoke=args.smoke,
        seed=args.seed,
    )

    print(
        "Loading PhoBERT tokenizer...",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
    )

    # ------------------------------------------
    # SEARCH SPACE
    # ------------------------------------------

    if args.smoke:

        configs = [
            {
                "learning_rate": 2e-5,
                "batch_size": 1,
                "gradient_accumulation_steps": 2,
                "max_length": 32,
                "weight_decay": 0.01,
            },
        ]

        epochs = 1

    else:

        # GPU RTX 2050 4GB:
        # Microbatch nho, effective batch = 8

        configs = [
            {
                "learning_rate": 2e-5,
                "batch_size": 1,
                "gradient_accumulation_steps": 8,
                "max_length": 64,
                "weight_decay": 0.01,
            },
            {
                "learning_rate": 3e-5,
                "batch_size": 1,
                "gradient_accumulation_steps": 8,
                "max_length": 64,
                "weight_decay": 0.01,
            },
            {
                "learning_rate": 2e-5,
                "batch_size": 2,
                "gradient_accumulation_steps": 4,
                "max_length": 64,
                "weight_decay": 0.01,
            },
            {
                "learning_rate": 3e-5,
                "batch_size": 2,
                "gradient_accumulation_steps": 4,
                "max_length": 64,
                "weight_decay": 0.01,
            },
        ]

        epochs = args.epochs

        # Mac dinh chay 2 cau hinh
        if not args.full:
            configs = configs[:2]

        # --trials N chi chay N cau hinh dau
        if args.trials is not None:
            configs = configs[
                      :args.trials
                      ]

    # ------------------------------------------
    # OUTPUT DIRECTORIES
    # ------------------------------------------

    output_root = (
            ROOT / "results"
    )

    checkpoint_dir = (
            ROOT / "checkpoints"
    )

    if args.smoke:
        output_root = (
                output_root / "smoke"
        )

        checkpoint_dir = (
                checkpoint_dir / "smoke"
        )

    metrics_dir = (
            output_root / "metrics"
    )

    history_dir = (
            output_root / "training_history"
    )

    cm_dir = (
            output_root / "confusion_matrix"
    )

    for directory in [
        metrics_dir,
        history_dir,
        cm_dir,
        checkpoint_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ------------------------------------------
    # CLASS WEIGHTS
    # ------------------------------------------

    class_counts = np.bincount(
        train["sentiment"].to_numpy(),
        minlength=3,
    )

    weights = len(train) / (
            3 * np.maximum(
        class_counts,
        1,
    )
    )

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    print(
        "Class counts:",
        class_counts.tolist(),
    )

    print(
        "Class weights:",
        weights.tolist(),
    )

    # ------------------------------------------
    # CACHE TOKENIZED DATA
    # ------------------------------------------

    datasets_cache = {}

    all_trials = []
    all_history = []

    global_best = -1.0
    global_params = None
    global_eval = None
    global_history = None

    # ------------------------------------------
    # TRIAL LOOP
    # ------------------------------------------

    for trial_id, config in enumerate(
            configs,
            start=1,
    ):
        print(
            f"\n===== PHOBERT "
            f"{trial_id}/{len(configs)} =====",
            flush=True,
        )

        print(
            "Parameters:",
            config,
            flush=True,
        )

        max_length = config[
            "max_length"
        ]

        if max_length not in datasets_cache:

            train_dataset = create_dataset(
                train,
                tokenizer,
                max_length,
            )

            val_dataset = create_dataset(
                val,
                tokenizer,
                max_length,
            )

            datasets_cache[max_length] = (
                train_dataset,
                val_dataset,
            )

        train_dataset, val_dataset = (
            datasets_cache[max_length]
        )

        try:
            result = train_one(
                config=config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                class_weights=class_weights,
                epochs=epochs,
                patience=args.patience,
                seed=args.seed,
                checkpoint_dir=checkpoint_dir,
                tokenizer=tokenizer,
                global_best=global_best,
            )

        except torch.OutOfMemoryError:
            print(
                "\nCUDA OUT OF MEMORY!"
            )
            print(
                "GPU khong du VRAM cho cau hinh nay."
            )
            print(
                "Hay giam batch_size hoac max_length."
            )
            raise

        # --------------------------------------
        # COLLECT TRIAL RESULTS
        # --------------------------------------

        all_trials.append({
            "trial": trial_id,
            **config,
            "effective_batch_size": (
                    config["batch_size"]
                    * config[
                        "gradient_accumulation_steps"
                    ]
            ),
            "best_epoch": result[
                "best_epoch"
            ],
            "val_macro_f1": result[
                "best_score"
            ],
            "val_neutral_f1": result[
                "best_eval"
            ]["neutral_f1"],
            "val_accuracy": result[
                "best_eval"
            ]["accuracy"],
        })

        for record in result["history"]:
            all_history.append({
                "trial": trial_id,
                **config,
                **record,
            })

        # Luu CSV sau moi trial
        pd.DataFrame(
            all_trials
        ).to_csv(
            metrics_dir / "phobert_trials.csv",
            index=False,
            encoding="utf-8-sig",
            )

        pd.DataFrame(
            all_history
        ).to_csv(
            history_dir / "phobert_history.csv",
            index=False,
            encoding="utf-8-sig",
            )

        print(
            f"BEST trial {trial_id}: "
            f"Macro F1="
            f"{result['best_score']:.4f} | "
            f"Epoch="
            f"{result['best_epoch']}",
            flush=True,
        )

        # --------------------------------------
        # GLOBAL BEST
        # --------------------------------------

        if result["best_score"] > global_best:

            global_best = result[
                "best_score"
            ]

            global_params = (
                config.copy()
            )

            global_params["best_epoch"] = (
                result["best_epoch"]
            )

            global_eval = result[
                "best_eval"
            ]

            global_history = result[
                "history"
            ]

            summary = {
                "model": "PhoBERT",
                "model_name": MODEL_NAME,
                "selection_metric": (
                    "validation_macro_f1"
                ),
                "best_validation_macro_f1": (
                    global_best
                ),
                "best_validation_neutral_f1": (
                    global_eval["neutral_f1"]
                ),
                "best_validation_accuracy": (
                    global_eval["accuracy"]
                ),
                "best_params": global_params,
                "seed": args.seed,
                "data_split": (
                    "smoke"
                    if args.smoke
                    else "full_train_validation"
                ),
                "search_trials_completed": (
                    trial_id
                ),
            }

            with open(
                    metrics_dir / "phobert_best.json",
                    "w",
                    encoding="utf-8",
            ) as f:

                json.dump(
                    summary,
                    f,
                    ensure_ascii=False,
                    indent=4,
                )

            save_confusion_matrix(
                global_eval,
                cm_dir / "phobert_validation.webp",
                )

            save_history_chart(
                global_history,
                history_dir
                / "phobert_best_history.webp",
                )

            report = classification_report(
                global_eval["y_true"],
                global_eval["y_pred"],
                labels=[0, 1, 2],
                target_names=LABEL_NAMES,
                zero_division=0,
            )

            with open(
                    metrics_dir
                    / "phobert_classification_report.txt",
                    "w",
                    encoding="utf-8",
            ) as f:

                f.write(report)

    # ------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------

    print(
        "\n========== FINAL RESULT ==========",
        flush=True,
    )

    print(
        "Best params:",
        global_params,
    )

    print(
        f"Best Validation Macro F1: "
        f"{global_best:.4f}"
    )

    print(
        "Best Validation Neutral F1:",
        round(
            global_eval["neutral_f1"],
            4,
        ),
    )

    print(
        "Saved results to:",
        output_root,
    )

    print(
        "Saved model to:",
        checkpoint_dir / "phobert_best",
        )

    print(
        "Test dataset was NOT used."
    )


if __name__ == "__main__":
    main()