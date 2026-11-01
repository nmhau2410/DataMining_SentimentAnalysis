from collections import Counter
from pathlib import Path
import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
RESULT_DIR = ROOT / "results" / "preprocessing"
MODEL_NAME = "vinai/phobert-base-v2"


def length_stats(lengths):
    lengths = np.array(lengths)

    return {
        "samples": int(len(lengths)),
        "min": int(lengths.min()),
        "mean": round(float(lengths.mean()), 2),
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
        "coverage_32": round(float((lengths <= 32).mean() * 100), 2),
        "coverage_64": round( float((lengths <= 64).mean() * 100), 2),
        "coverage_128": round(float((lengths <= 128).mean() * 100), 2),
    }


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    texts = (train["text_segmented"].fillna("").astype(str))

    # TEXTCNN TOKEN LENGTH
    word_lengths = texts.str.split().str.len()
    textcnn_stats = length_stats(word_lengths)

    with open(RESULT_DIR / "textcnn_length_stats.json", "w", encoding="utf-8") as f:
        json.dump(textcnn_stats, f, ensure_ascii=False, indent=4)

    # PHOBERT TOKEN LENGTH
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,use_fast=False)
    encoded = tokenizer(
        texts.tolist(),
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )

    phobert_lengths = [len(ids) for ids in encoded["input_ids"]]
    phobert_stats = length_stats(phobert_lengths)

    with open( RESULT_DIR / "phobert_length_stats.json", "w", encoding="utf-8",) as f:
        json.dump(phobert_stats, f, ensure_ascii=False, indent=4)

    # VOCABULARY EVIDENCE
    counter = Counter()

    for text in texts:
        counter.update(text.lower().split())

    total_tokens = sum(counter.values())
    vocab_rows = []

    for min_freq in [1, 2, 3, 5]:
        kept = {
            word: freq
            for word, freq in counter.items()
            if freq >= min_freq
        }

        kept_tokens = sum(kept.values())
        vocab_rows.append({
            "min_freq": min_freq,
            "vocab_size": len(kept),
            "token_coverage_percent": round(kept_tokens / total_tokens * 100, 2)
        })

    pd.DataFrame(vocab_rows).to_csv(
        RESULT_DIR / "vocabulary_stats.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # CLASS WEIGHTS EVIDENCE
    counts = (train["sentiment"].value_counts().sort_index())
    label_names = {
        0: "negative",
        1: "neutral",
        2: "positive",
    }
    weights = []

    for label, count in counts.items():
        weight = len(train) / (3 * count)
        weights.append({
            "label": int(label),
            "label_name": label_names[int(label)],
            "count": int(count),
            "weight": float(weight),
        })

    pd.DataFrame(weights).to_csv(RESULT_DIR / "class_weights.csv", index=False, encoding="utf-8-sig")

    print("\n===== TEXTCNN LENGTH =====")
    print(json.dumps(textcnn_stats, indent=4, ensure_ascii=False))

    print("\n===== PHOBERT LENGTH =====")
    print(json.dumps(phobert_stats, indent=4, ensure_ascii=False))

    print("\n===== VOCABULARY =====")
    print(pd.DataFrame(vocab_rows).to_string(index=False))

    print("\n===== CLASS WEIGHTS =====")
    print(pd.DataFrame(weights).to_string(index=False))

    print("\nEvidence saved to:")
    print(RESULT_DIR)


if __name__ == "__main__":
    main()