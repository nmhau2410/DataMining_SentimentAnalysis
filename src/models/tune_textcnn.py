import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
RESULT_DIR = ROOT / "results" / "textcnn_tuning_weighted"

SEED = 42
CONFIRM_SEEDS = [123, 777, 2026]

N_TRIALS = 120
TOP_K = 5
MAX_EPOCHS = 50
MAX_LENGTH = 64
MIN_FREQ = 2

KERNEL_OPTIONS = {
    "123": [1, 2, 3],
    "234": [2, 3, 4],
    "345": [3, 4, 5],
    "456": [4, 5, 6],
    "567": [5, 6, 7],
    "2345": [2, 3, 4, 5],
    "3456": [3, 4, 5, 6]
}


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

    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    _, _, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average=None,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "neutral_f1": f1[1]
    }


def build_loaders(train_dataset, valid_dataset, batch_size, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, valid_loader


def train_model(params, train_dataset, valid_dataset, class_weights, vocab_size, device, seed, trial=None):
    set_seed(seed)

    train_loader, valid_loader = build_loaders(
        train_dataset,
        valid_dataset,
        params["batch_size"],
        seed
    )

    model = TextCNN(
        vocab_size=vocab_size,
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

    best_epoch = 0
    best_macro_f1 = -1
    best_metrics = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

        metrics = evaluate(model, valid_loader, device)

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_epoch = epoch
            best_metrics = metrics.copy()

        if trial is not None:
            trial.report(metrics["macro_f1"], epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

    del model
    torch.cuda.empty_cache()

    return {
        "best_epoch": best_epoch,
        **best_metrics
    }


def get_trial_params(trial):
    return {
        "embedding_dim": trial.params["embedding_dim"],
        "num_filters": trial.params["num_filters"],
        "kernel_sizes": KERNEL_OPTIONS[trial.params["kernels"]],
        "dropout": trial.params["dropout"],
        "lr": trial.params["lr"],
        "weight_decay": trial.params["weight_decay"],
        "batch_size": trial.params["batch_size"]
    }


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Imbalance strategy: weighted_ce")
    print("Max length:", MAX_LENGTH)
    print("Min freq:", MIN_FREQ)

    train = pd.read_csv(DATA_DIR / "train.csv")
    valid = pd.read_csv(DATA_DIR / "validation.csv")

    labels = train["sentiment"].astype(int).to_numpy()

    print("Train samples:", len(train))
    print(train["sentiment"].value_counts().sort_index())
    print("Validation samples:", len(valid))
    print(valid["sentiment"].value_counts().sort_index())

    vocab = build_vocab(train["text_segmented"], MIN_FREQ)

    print("Vocabulary:", len(vocab))

    train_dataset = encode_data(train, vocab)
    valid_dataset = encode_data(valid, vocab)
    class_weights = get_class_weights(labels, device)

    print("Class weights:", class_weights.cpu().numpy())

    def objective(trial):
        params = {
            "embedding_dim": trial.suggest_categorical("embedding_dim", [64, 128, 200, 300]),
            "num_filters": trial.suggest_categorical("num_filters", [64, 100, 128, 200]),
            "kernel_sizes": KERNEL_OPTIONS[trial.suggest_categorical("kernels", list(KERNEL_OPTIONS.keys()))],
            "dropout": trial.suggest_float("dropout", 0.2, 0.6, step=0.1),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2]),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        }

        result = train_model(
            params=params,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            class_weights=class_weights,
            vocab_size=len(vocab),
            device=device,
            seed=SEED,
            trial=trial
        )

        trial.set_user_attr("best_epoch", result["best_epoch"])
        trial.set_user_attr("accuracy", result["accuracy"])
        trial.set_user_attr("neutral_f1", result["neutral_f1"])

        return result["macro_f1"]

    storage = f"sqlite:///{(RESULT_DIR / 'optuna.db').as_posix()}"

    sampler = optuna.samplers.TPESampler(seed=SEED)

    pruner = optuna.pruners.HyperbandPruner(
        min_resource=5,
        max_resource=MAX_EPOCHS,
        reduction_factor=3
    )

    study = optuna.create_study(
        study_name="textcnn_weighted_ce",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True
    )

    remaining_trials = max(0, N_TRIALS - len(study.trials))

    print("\nExisting trials:", len(study.trials))
    print("Remaining trials:", remaining_trials)

    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials)

    trials = []

    for trial in study.trials:
        trials.append({
            "trial": trial.number,
            "state": trial.state.name,
            "macro_f1": trial.value,
            "best_epoch": trial.user_attrs.get("best_epoch"),
            "accuracy": trial.user_attrs.get("accuracy"),
            "neutral_f1": trial.user_attrs.get("neutral_f1"),
            **trial.params
        })

    pd.DataFrame(trials).to_csv(
        RESULT_DIR / "trials.csv",
        index=False,
        encoding="utf-8-sig"
    )

    completed = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]

    completed = sorted(
        completed,
        key=lambda trial: trial.value,
        reverse=True
    )

    top_trials = completed[:TOP_K]

    print("\n" + "=" * 70)
    print("TOP TRIALS")
    print("=" * 70)

    for trial in top_trials:
        print(
            f"Trial {trial.number} | "
            f"Macro F1 {trial.value:.6f} | "
            f"Neutral F1 {trial.user_attrs.get('neutral_f1', 0):.6f} | "
            f"Epoch {trial.user_attrs.get('best_epoch')}"
        )

    confirmation_path = RESULT_DIR / "confirmation_runs.csv"

    if confirmation_path.exists():
        confirmation = pd.read_csv(confirmation_path)
    else:
        confirmation = pd.DataFrame()

    confirmation_rows = confirmation.to_dict("records") if not confirmation.empty else []

    for trial in top_trials:
        params = get_trial_params(trial)

        for seed in CONFIRM_SEEDS:
            already_done = False

            if not confirmation.empty:
                mask = (
                    (confirmation["trial"] == trial.number) &
                    (confirmation["seed"] == seed)
                )
                already_done = mask.any()

            if already_done:
                print(f"Skip confirmation Trial {trial.number}, seed {seed}")
                continue

            print("\n" + "-" * 70)
            print("Confirmation trial:", trial.number)
            print("Seed:", seed)

            result = train_model(
                params=params,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                class_weights=class_weights,
                vocab_size=len(vocab),
                device=device,
                seed=seed
            )

            row = {
                "trial": trial.number,
                "seed": seed,
                "best_epoch": result["best_epoch"],
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "neutral_f1": result["neutral_f1"],
                **trial.params
            }

            confirmation_rows.append(row)

            confirmation = pd.DataFrame(confirmation_rows)

            confirmation.to_csv(
                confirmation_path,
                index=False,
                encoding="utf-8-sig"
            )

            print("Best epoch:", result["best_epoch"])
            print("Macro F1:", round(result["macro_f1"], 6))
            print("Neutral F1:", round(result["neutral_f1"], 6))

    confirmation = pd.DataFrame(confirmation_rows)

    summary = confirmation.groupby("trial").agg(
        mean_macro_f1=("macro_f1", "mean"),
        std_macro_f1=("macro_f1", "std"),
        mean_neutral_f1=("neutral_f1", "mean"),
        mean_accuracy=("accuracy", "mean"),
        epoch_1=("best_epoch", "min"),
        epoch_3=("best_epoch", "max")
    ).reset_index()

    median_epochs = confirmation.groupby("trial")["best_epoch"].median().reset_index()
    median_epochs = median_epochs.rename(columns={"best_epoch": "median_best_epoch"})

    summary = summary.merge(median_epochs, on="trial")

    summary = summary.sort_values(
        ["mean_macro_f1", "mean_neutral_f1", "std_macro_f1"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    summary.to_csv(
        RESULT_DIR / "confirmation_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 70)
    print("CONFIRMATION SUMMARY")
    print("=" * 70)

    print(
        summary[
            [
                "trial",
                "mean_macro_f1",
                "std_macro_f1",
                "mean_neutral_f1",
                "mean_accuracy",
                "median_best_epoch"
            ]
        ].to_string(index=False)
    )

    winner_trial_number = int(summary.iloc[0]["trial"])
    winner_trial = next(trial for trial in top_trials if trial.number == winner_trial_number)
    winner_params = get_trial_params(winner_trial)

    winner_runs = confirmation[confirmation["trial"] == winner_trial_number]
    final_epochs = int(winner_runs["best_epoch"].median())

    selected = {
        "trial": winner_trial_number,
        "strategy": "weighted_ce",
        "mean_macro_f1": float(summary.iloc[0]["mean_macro_f1"]),
        "std_macro_f1": float(summary.iloc[0]["std_macro_f1"]),
        "mean_neutral_f1": float(summary.iloc[0]["mean_neutral_f1"]),
        "mean_accuracy": float(summary.iloc[0]["mean_accuracy"]),
        "confirmation_seeds": CONFIRM_SEEDS,
        "confirmation_best_epochs": winner_runs["best_epoch"].astype(int).tolist(),
        "final_epochs": final_epochs,
        "params": winner_params,
        "fixed": {
            "max_length": MAX_LENGTH,
            "min_freq": MIN_FREQ,
            "optimizer": "AdamW",
            "loss": "WeightedCrossEntropy",
            "max_epochs": MAX_EPOCHS
        },
        "test_used": False
    }

    with open(RESULT_DIR / "selected_config.json", "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=4)

    print("\n" + "=" * 70)
    print("SELECTED CONFIG")
    print("=" * 70)

    print("Trial:", selected["trial"])
    print("Mean Macro F1:", round(selected["mean_macro_f1"], 6))
    print("Std Macro F1:", round(selected["std_macro_f1"], 6))
    print("Mean Neutral F1:", round(selected["mean_neutral_f1"], 6))
    print("Best epochs:", selected["confirmation_best_epochs"])
    print("Final epochs:", selected["final_epochs"])
    print("Params:", selected["params"])
    print("Test used:", selected["test_used"])

    print("\nSaved:")
    print(RESULT_DIR / "optuna.db")
    print(RESULT_DIR / "trials.csv")
    print(RESULT_DIR / "confirmation_runs.csv")
    print(RESULT_DIR / "confirmation_summary.csv")
    print(RESULT_DIR / "selected_config.json")


if __name__ == "__main__":
    main()