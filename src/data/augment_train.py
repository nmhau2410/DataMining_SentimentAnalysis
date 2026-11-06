import argparse
import random
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "augmented"

MIN_TOKENS = 6
MAX_SWAP_DISTANCE = 2

PROTECTED_WORDS = {
    "không", "chưa", "chẳng", "chả", "đừng",
    "rất", "quá", "hơi", "khá", "cực_kỳ",
    "tốt", "hay", "ổn", "tệ", "kém", "dở",
    "thích", "ghét", "hài_lòng",
    "có", "của", "và", "nhưng", "tuy_nhiên"
}


def is_candidate(token):
    token = token.lower()
    if token in PROTECTED_WORDS:
        return False
    if token.startswith("không_") or token.startswith("chưa_"):
        return False
    return any(char.isalnum() for char in token)


def random_swap(text, rng):
    tokens = str(text).split()
    if len(tokens) < MIN_TOKENS:
        return text

    candidates = [i for i, token in enumerate(tokens) if is_candidate(token)]
    if len(candidates) < 2:
        return text

    i = rng.choice(candidates)
    nearby = [j for j in candidates if j != i and abs(i - j) <= MAX_SWAP_DISTANCE]
    if not nearby:
        return text

    j = rng.choice(nearby)
    tokens[i], tokens[j] = tokens[j], tokens[i]
    return " ".join(tokens)


def random_delete(text, rng):
    tokens = str(text).split()
    if len(tokens) < 8:
        return text

    candidates = [
        i for i, token in enumerate(tokens)
        if is_candidate(token) and i != 0 and i != len(tokens) - 1
    ]
    if not candidates:
        return text

    index = rng.choice(candidates)
    del tokens[index]
    return " ".join(tokens)


def valid_augmentation(original, augmented):
    if original == augmented:
        return False

    original_tokens = original.split()
    augmented_tokens = augmented.split()

    if len(augmented_tokens) < MIN_TOKENS:
        return False
    if len(augmented_tokens) < len(original_tokens) * 0.8:
        return False
    return True


def augment_text(text, rng):
    if rng.random() < 0.7:
        new_text = random_swap(text, rng)
        aug_type = "swap"
    else:
        new_text = random_delete(text, rng)
        aug_type = "delete"

    if not valid_augmentation(str(text), new_text):
        return text, "none"
    return new_text, aug_type


def build_augmented_train(train, ratio, forbidden_txts, seed=42):
    rng = random.Random(seed)
    counts = train["sentiment"].value_counts()
    major_count = counts.max()

    neutral = train[train["sentiment"] == 1]
    target_neutral = round(major_count * ratio)
    target_neutral = max(target_neutral, len(neutral))

    res = train.copy()
    res["is_augmented"] = False
    res["augmentation_type"] = "original"
    res["source_text"] = res["text_segmented"]

    existing = set(train["text_segmented"].fillna("").astype(str).tolist())
    existing.update(forbidden_txts)

    augmented_rows = []
    attempts = 0
    max_attempts = 200000

    while len(neutral) + len(augmented_rows) < target_neutral:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("Can't gen enough unique augmented samples")

        src = neutral.iloc[rng.randrange(len(neutral))].copy()
        original_txt = str(src["text_segmented"])
        new_txt, aug_type = augment_text(original_txt, rng)

        if aug_type == "none":
            continue
        if not new_txt or new_txt in existing:
            continue

        existing.add(new_txt)
        readable = new_txt.replace("_", " ")

        src["source_text"] = original_txt
        src["text_segmented"] = new_txt
        src["text_normalized"] = readable
        src["sentence"] = readable
        src["is_augmented"] = True
        src["augmentation_type"] = aug_type
        augmented_rows.append(src)

    if augmented_rows:
        res = pd.concat([res, pd.DataFrame(augmented_rows)], ignore_index=True)

    res = res.sample(frac=1, random_state=seed).reset_index(drop=True)
    return res


def validate_augmented(data, valid, test):
    augmented = data[data["is_augmented"] == True]

    assert (augmented["sentiment"] == 1).all(), "Found augmented sample outside Neutral class"
    assert not augmented["text_segmented"].isna().any(), "Found empty augmented text"
    assert not augmented["text_segmented"].duplicated().any(), "Found duplicated augmented text"

    valid_txts = set(valid["text_segmented"].fillna("").astype(str))
    test_txts = set(test["text_segmented"].fillna("").astype(str))
    augmented_txts = set(augmented["text_segmented"].astype(str))

    assert len(augmented_txts & valid_txts) == 0, "Found augmented text duplicated with Validation"
    assert len(augmented_txts & test_txts) == 0, "Found augmented text duplicated with Test"

    print("Validation passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", nargs="+", type=float, default=[0.2, 0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_DIR / "train.csv")
    valid = pd.read_csv(DATA_DIR / "validation.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    forbidden_txts = set(valid["text_segmented"].fillna("").astype(str))
    forbidden_txts.update(test["text_segmented"].fillna("").astype(str))

    print("Original distribution:")
    print(train["sentiment"].value_counts().sort_index())

    summary = []

    for ratio in args.ratio:
        augmented = build_augmented_train(train=train, ratio=ratio, forbidden_txts=forbidden_txts, seed=args.seed)
        validate_augmented(augmented, valid, test)

        ratio_name = int(ratio * 100)
        output_path = OUTPUT_DIR / f"augmented_{ratio_name}.csv"
        augmented.to_csv(output_path, index=False, encoding="utf-8-sig")

        counts = augmented["sentiment"].value_counts().sort_index()
        augmented_count = int(augmented["is_augmented"].sum())

        summary.append({
            "ratio": ratio,
            "negative": int(counts.get(0, 0)),
            "neutral": int(counts.get(1, 0)),
            "positive": int(counts.get(2, 0)),
            "augmented_samples": augmented_count,
            "total_samples": len(augmented)
        })

        print(f"\nRatio: {ratio}")
        print(counts)
        print("Augmented:", augmented_count)
        print("Saved:", output_path)

    pd.DataFrame(summary).to_csv(
        OUTPUT_DIR / "augmentation_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )


if __name__ == "__main__":
    main()