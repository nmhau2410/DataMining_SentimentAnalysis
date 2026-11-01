from src.data.load_data import load_uit_vsfc
from pathlib import Path
import json
import pandas as pd
from src.data.load_data import load_uit_vsfc

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT_DIR / "results" / "preprocessing"

LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

def audit_split(name, df):
    print(f"\n===== {name.upper()} =====")
    print("Samples:", len(df))

    missing_sentence = int(df["sentence"].isnull().sum())
    missing_sentiment = int(df["sentiment"].isnull().sum())

    print("\nMissing values:")
    print("sentence:", missing_sentence)
    print("sentiment:", missing_sentiment)

    empty = int(df["sentence"].fillna("").str.strip().eq("").sum())
    print("\nEmpty sentences:", empty)

    duplicate = int( df["sentence"].duplicated().sum())
    print("Duplicate sentences:", duplicate)

    counts = ( df["sentiment"].value_counts().sort_index())
    print("\nSentiment distribution:")

    distribution = []

    for label, count in counts.items():
        percent = count / len(df) * 100

        print(
            f"{LABEL_MAP[label]}: "
            f"{count} ({percent:.2f}%)"
        )

        distribution.append({
            "split": name,
            "label": int(label),
            "label_name": LABEL_MAP[label],
            "count": int(count),
            "percent": round(percent, 2),
        })

    summary = {
        "split": name,
        "samples": len(df),
        "missing_sentence": missing_sentence,
        "missing_sentiment": missing_sentiment,
        "empty_sentence": empty,
        "duplicate_sentence": duplicate,
    }

    return summary, distribution

def check_cross_split_duplicates(train_df, val_df,test_df,):
    print("\n===== CROSS-SPLIT DUPLICATES =====")
    train = set(train_df["sentence"])
    val = set(val_df["sentence"])
    test = set(test_df["sentence"])

    result = {
        "train_validation": len(train & val),
        "train_test": len(train & test),
        "validation_test": len(val & test),
    }

    print( "Train - Validation:",result["train_validation"],)
    print("Train - Test:",result["train_test"],)
    print( "Validation - Test:",result["validation_test"],)
    return result

def main():
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = load_uit_vsfc()

    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()

    summaries = []
    distributions = []

    for name, df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        summary, distribution = audit_split(
            name,
            df,
        )

        summaries.append(summary)
        distributions.extend(distribution)

    pd.DataFrame(summaries).to_csv(
        RESULT_DIR / "audit_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(distributions).to_csv(
        RESULT_DIR / "class_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cross_duplicates = (
        check_cross_split_duplicates(
            train_df,
            val_df,
            test_df,
        )
    )

    with open(
        RESULT_DIR / "cross_split_duplicates.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            cross_duplicates,
            f,
            ensure_ascii=False,
            indent=4,
        )

    duplicates = train_df[
        train_df.duplicated(
            subset="sentence",
            keep=False,
        )
    ]

    if len(duplicates) > 0:
        print("\n===== TRAIN DUPLICATE DETAILS =====")
        print(duplicates[["sentence", "sentiment", "topic"]].to_string(index=False))

        duplicates[["sentence", "sentiment", "topic"]].to_csv(
            RESULT_DIR / "duplicate_samples.csv",
            index=False,
            encoding="utf-8-sig",
        )

        conflicts = (duplicates.groupby("sentence")["sentiment"].nunique())
        conflict_sentences = conflicts[ conflicts > 1].index
        conflict_rows = duplicates[duplicates["sentence"].isin(  conflict_sentences)]
        conflict_rows.to_csv(
            RESULT_DIR / "conflicting_samples.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("\nEvidence saved to:",RESULT_DIR,)

if __name__ == "__main__":
    main()