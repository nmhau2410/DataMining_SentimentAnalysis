import argparse
import json
import random
from collections import Counter
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
import torch
from optuna.pruners import HyperbandPruner, NopPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
RESULT_DIR = ROOT / "results" / "textcnn_tuning_v2"
CONFIG_DIR = ROOT / "configs"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 64
MIN_FREQ = 2
LABELS = [0, 1, 2]

KERNEL_CHOICES = {
    "123": (1, 2, 3),
    "234": (2, 3, 4),
    "345": (3, 4, 5),
    "456": (4, 5, 6),
    "567": (5, 6, 7),
    "2345": (2, 3, 4, 5),
    "3456": (3, 4, 5, 6),
}

CONFIRM_SEEDS = [123, 777, 2026]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 2. DOC DU LIEU

def load_data(smoke=False):
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "validation.csv")

    if smoke:
        train = train.sample(n=min(512, len(train)), random_state=42)
        val = val.sample(n=min(256, len(val)), random_state=42)

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)

    print("Train:", len(train))
    print("Validation:", len(val))

    return train, val


# 3. TAO VOCABULARY

def build_vocab(texts,min_freq=MIN_FREQ):
    counter = Counter()

    for text in texts:
        counter.update(str(text).lower().split())

    vocab = {"<PAD>": 0, "<UNK>": 1}

    for word in sorted(counter):
        if counter[word] >= min_freq:
            vocab[word] = len(vocab)

    return vocab


def encode_texts(texts, vocab, max_length=MAX_LENGTH,):
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

def make_loaders(X_train, y_train, X_val, y_val, batch_size, seed,):
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val,),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader


def make_criterion(y_train, class_weight,):
    if class_weight == "none":
        return nn.CrossEntropyLoss()

    counts = np.bincount(y_train.numpy(),minlength=3)
    weights = len(y_train) / (3 * np.maximum(counts, 1))
    weights = torch.tensor(weights, dtype=torch.float32,device=DEVICE)

    return nn.CrossEntropyLoss(weight=weights)


def train_config(config, X_train, y_train, X_val, y_val, vocab_size, epochs, seed, trial=None,):
    set_seed(seed)

    train_loader, val_loader = (
        make_loaders(X_train, y_train, X_val, y_val, config["batch_size"], seed,)
    )

    model = TextCNN(
        vocab_size=vocab_size,
        embedding_dim=(config["embedding_dim"]),
        num_filters=(config["num_filters"]),
        kernel_sizes=(config["kernel_sizes"]),
        dropout=config["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=(config["weight_decay"]),
    )

    criterion = make_criterion(y_train, config["class_weight"])

    best = None

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_items = 0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y,)
            loss.backward()
            optimizer.step()
            total_loss += (loss.item() * len(y))
            total_items += len(y)

        metrics = evaluate(model, val_loader, criterion)

        result = {
            "epoch": epoch,
            "train_loss": total_loss / total_items, **metrics,
        }

        if (
            best is None
            or result["macro_f1"]
            > best["macro_f1"]
            or (
                result["macro_f1"]
                == best["macro_f1"]
                and result["neutral_f1"]
                > best["neutral_f1"]
            )
        ):
            best = result

        if trial is not None:
            trial.report(result["macro_f1"], step=epoch,)

            if trial.should_prune():
                raise optuna.TrialPruned()

    return best

def suggest_config(trial):
    kernel_name = (trial.suggest_categorical("kernel_sizes", list(KERNEL_CHOICES)))

    return {
        "embedding_dim": trial.suggest_categorical("embedding_dim", [64, 128, 200, 300],),
        "num_filters": trial.suggest_categorical("num_filters", [64, 100, 128, 200],),
        "kernel_sizes": KERNEL_CHOICES[kernel_name],
        "dropout": trial.suggest_float("dropout", 0.2, 0.6, step=0.1,),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True,),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128],),
        "class_weight": trial.suggest_categorical("class_weight", ["none", "balanced"]),
    }


def objective(trial, X_train, y_train, X_val, y_val, vocab_size, epochs, seed,):
    config = suggest_config(trial)

    best = train_config(
        config=config,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        vocab_size=vocab_size,
        epochs=epochs,
        seed=seed,
        trial=trial,
    )

    trial.set_user_attr("best_epoch", best["epoch"])
    trial.set_user_attr("neutral_f1", best["neutral_f1"])
    trial.set_user_attr("accuracy", best["accuracy"])
    return best["macro_f1"]


def trial_to_config(trial):
    params = trial.params

    return {
        "embedding_dim": params["embedding_dim"],
        "num_filters": params["num_filters"],
        "kernel_sizes": KERNEL_CHOICES[params["kernel_sizes"]],
        "dropout": params["dropout"],
        "learning_rate": params["learning_rate"],
        "weight_decay": params["weight_decay"],
        "batch_size": params["batch_size"],
        "class_weight": params["class_weight"],
    }


def confirm_top_configs(study, X_train, y_train, X_val, y_val, vocab_size, epochs, top_k):
    completed = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    completed.sort(key=lambda trial: trial.value,reverse=True)

    top_trials = completed[:top_k]

    output_path = (RESULT_DIR / "confirmation_trials.csv")
    if output_path.exists():
        rows = pd.read_csv(output_path).to_dict("records")
    else:
        rows = []

    for trial in top_trials:
        config = trial_to_config(trial)

        print("\nConfirm trial:", trial.number,)

        for seed in CONFIRM_SEEDS:
            already_done = any(
                int(row["trial"]) == trial.number
                and int(row["seed"]) == seed
                for row in rows
            )

            if already_done:
                print(
                    f"Skip trial "
                    f"{trial.number}, "
                    f"seed {seed}"
                )
                continue

            best = train_config(
                config=config,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                vocab_size=vocab_size,
                epochs=epochs,
                seed=seed,
            )

            rows.append({
                "trial": trial.number,
                "seed": seed,
                **config,
                "best_epoch": best["epoch"],
                "macro_f1": best["macro_f1"],
                "neutral_f1": best["neutral_f1"],
                "accuracy": best["accuracy"],
            })

            pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig",)

    current_trials = {trial.number for trial in top_trials}

    df = pd.DataFrame(rows)
    return df[df["trial"].isin(current_trials)].copy()


def select_best(confirmation):
    summary = (confirmation.groupby("trial").agg(
            mean_macro_f1=("macro_f1", "mean",),
            std_macro_f1=("macro_f1", "std",),
            mean_neutral_f1=("neutral_f1", "mean",),
            mean_accuracy=("accuracy", "mean",),
        ).reset_index()
    )

    summary = summary.sort_values(
        [
            "mean_macro_f1",
            "mean_neutral_f1",
            "std_macro_f1",
        ],
        ascending=[False, False, True,],
    )

    best_trial = int(summary.iloc[0]["trial"])
    return best_trial, summary

# 7. TIM BO THAM SO TOT NHAT
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",)
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=50,)
    parser.add_argument("--top-k", type=int, default=5,)
    parser.add_argument("--seed", type=int, default=42,)

    args = parser.parse_args()

    set_seed(args.seed)

    if DEVICE.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))

    train, val = load_data(smoke=args.smoke)

    train_texts = (train["text_segmented"].fillna("").astype(str))
    val_texts = (val["text_segmented"].fillna("").astype(str))

    vocab = build_vocab(train_texts)

    X_train = encode_texts(train_texts,vocab)
    X_val = encode_texts(val_texts, vocab,)
    y_train = torch.tensor(train["sentiment"].to_numpy(), dtype=torch.long)
    y_val = torch.tensor(val["sentiment"].to_numpy(), dtype=torch.long,)

    RESULT_DIR.mkdir(parents=True, exist_ok=True,)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True,)

    print("Device:", DEVICE)
    print("Train:", len(train))
    print("Validation:", len(val))
    print("Vocabulary:", len(vocab))
    print("Max length:", MAX_LENGTH)
    print("Min freq:", MIN_FREQ)

    epochs = (1 if args.smoke else args.epochs)
    sampler = TPESampler(seed=args.seed)

    if args.smoke:
        pruner = NopPruner()
        storage = None
        study_name = None
    else:
        pruner = HyperbandPruner(min_resource=5, max_resource=args.epochs, reduction_factor=3)
        db_path = (RESULT_DIR / "optuna.db").resolve()
        storage = ("sqlite:///" + str(db_path).replace("\\", "/"))
        study_name = ("textcnn_tuning_v2")

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=(
            not args.smoke
        ),
    )

    if args.smoke:
        remaining_trials = 1
    else:
        finished_trials = sum(trial.state in {
                TrialState.COMPLETE,
                TrialState.PRUNED,
            }
            for trial in study.trials
        )

        remaining_trials = max(0, args.trials - finished_trials,)

    print(
        f"Finished trials: "
        f"{len(study.trials)}"
    )

    print(
        f"Remaining trials: "
        f"{remaining_trials}"
    )

    if remaining_trials > 0:
        study.optimize(
            lambda trial: objective(
                trial=trial,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                vocab_size=len(vocab),
                epochs=epochs,
                seed=args.seed,
            ),
            n_trials=remaining_trials,
        )
    else:
        print("Search already completed.")

    if args.smoke:
        print("\nSmoke test completed.")
        return

    search_df = study.trials_dataframe()
    search_df.to_csv(RESULT_DIR / "search_trials.csv", index=False, encoding="utf-8-sig",)

    confirmation = (
        confirm_top_configs(
            study=study,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            vocab_size=len(vocab),
            epochs=args.epochs,
            top_k=args.top_k,
        )
    )

    confirmation.to_csv(RESULT_DIR / "confirmation_trials.csv", index=False, encoding="utf-8-sig",)

    best_trial_number, summary = (select_best( confirmation))
    summary.to_csv(RESULT_DIR / "confirmation_summary.csv",index=False, encoding="utf-8-sig",)

    best_trial = next(trial for trial in study.trials if trial.number == best_trial_number)
    best_config = (trial_to_config(best_trial))

    selected = {
        "embedding_dim": best_config["embedding_dim"],
        "num_filters": best_config["num_filters"],
        "kernel_sizes": list(best_config["kernel_sizes"]),
        "dropout": best_config["dropout"],
        "learning_rate": best_config["learning_rate"],
        "weight_decay":best_config["weight_decay"],
        "batch_size": best_config["batch_size"],
        "class_weight": best_config["class_weight"],
        "max_length": MAX_LENGTH,
        "min_freq": MIN_FREQ,
        "optimizer": "AdamW",
        "epochs": args.epochs,
        "selection_metric": "mean_validation_macro_f1",
        "confirmation_seeds": CONFIRM_SEEDS,
    }

    best_row = (summary[summary["trial"] == best_trial_number].iloc[0])

    evidence = {
        "model": "TextCNN",
        "search_method": "Optuna TPE + Hyperband",
        "search_trials": len(study.trials),
        "best_trial": best_trial_number,
        "selection_metric": "mean_validation_macro_f1",
        "tie_breakers": ["mean_validation_neutral_f1", "lower_std_macro_f1"],
        "mean_validation_macro_f1": float(best_row["mean_macro_f1"]),
        "std_validation_macro_f1": float(best_row["std_macro_f1"]),
        "mean_validation_neutral_f1": float(best_row["mean_neutral_f1"]),
        "best_params": selected,
        "test_used": False,

        "search_space": {
            "embedding_dim": [64, 128, 200, 300],
            "num_filters": [64, 100, 128, 200],
            "kernel_sizes": {key: list(value) for key, value in KERNEL_CHOICES.items()},
            "dropout": [0.2, 0.3, 0.4, 0.5, 0.6],
            "learning_rate": {
                "low": 1e-4,
                "high": 3e-3,
                "log": True,
            },
            "weight_decay": [0.0, 1e-5, 1e-4, 1e-3, 1e-2,],
            "batch_size": [16, 32, 64, 128],
            "class_weight": ["none", "balanced"],
        },

        "fixed_params": {
            "max_length": MAX_LENGTH,
            "min_freq": MIN_FREQ,
            "optimizer": "AdamW",
            "activation": "ReLU",
            "pooling": "global_max_pooling",
            "max_epochs": args.epochs,
        },
    }

    with open(RESULT_DIR / "textcnn_best.json", "w", encoding="utf-8",) as f:
        json.dump(evidence, f, ensure_ascii=False, indent=4)

    selected_path = (CONFIG_DIR / "selected_hyperparameters.json")
    if selected_path.exists():
        all_selected = json.loads(selected_path.read_text(encoding="utf-8"))
    else:
        all_selected = {}

    all_selected["textcnn"] = selected

    selected_path.write_text(json.dumps(all_selected, ensure_ascii=False, indent=4), encoding="utf-8",)

    print("\n========== BEST ==========")
    print(json.dumps(selected, ensure_ascii=False, indent=4))
    print( "\nMean Validation Macro F1:",round(float(best_row["mean_macro_f1"]), 4))
    print("Test dataset was NOT used.")
    print("Selected config:",selected_path,)

if __name__ == "__main__":
    main()