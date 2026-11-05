import argparse
import random
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "augmented"
PROTECTED_WORDS = {
    "không", "chưa", "chẳng", "chả", "đừng",
    "rất", "quá", "hơi", "khá", "cực_kỳ",
    "tốt", "hay", "ổn", "tệ", "kém", "dở",
    "thích", "ghét", "hài_lòng",
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
    candidates = [i for i, token in enumerate(tokens) if is_candidate(token)]
    if len(candidates) < 2:
        return text

    i, j = rng.sample(candidates, 2)
    tokens[i], tokens[j] = tokens[j], tokens[i]
    return " ".join(tokens)


def random_delete(text, rng):
    tokens = str(text).split()

    if len(tokens) <= 4:
        return text

    candidates = [i for i, token in enumerate(tokens) if is_candidate(token)]

    if not candidates:
        return text
    index = rng.choice(candidates)
    del tokens[index]
    return " ".join(tokens)


def augment_text(text, rng):
    if rng.random() < 0.7:
        return random_swap(text, rng), "swap"

    return random_delete(text, rng), "delete"


def build_augmented_train(train, ratio, forbidden_txts, seed=42):
    rng = random.Random(seed)
    counts = train["sentiment"].value_counts()
    major_counts = counts.max()

    neutral = train[train["sentiment"] == 1]
    target_neutral = round(major_counts * ratio)
    target_neutral = max(target_neutral, len(neutral))

    res = train.copy()
    res["is_augmented"] = False
    res["augmentation_type"] = "original"

    existing = set(train["text_segmented"].fillna("").astype(str).tolist())
    augmented_rows = []
    attempts = 0
    max_attempts = 200000

    while len(neutral) + len(augmented_rows) < target_neutral:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("Can't gen enough unique augmented samples")

        src = neutral.iloc[rng.randrange(len(neutral))].copy()

        new_txt, aug_type = augment_text(src["text_segmented"], rng)
        if not new_txt or new_txt in existing:
            continue

        existing.add(new_txt)
        readable = new_txt.replace("_", " ")

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
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
