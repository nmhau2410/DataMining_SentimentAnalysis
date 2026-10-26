from src.data.load_data import load_uit_vsfc

LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

def audit_split(name, df):
    print(f"\n===== {name.upper()} =====")
    print("Samples:", len(df))

    print("\nMissing values:")
    print(df[["sentence", "sentiment"]].isnull().sum())

    empty = df["sentence"].fillna("").str.strip().eq("").sum()
    print("\nEmpty sentences:", empty)

    duplicate = df["sentence"].duplicated().sum()
    print("Duplicate sentences:", duplicate)

    print("\nSentiment distribution:")
    counts = df["sentiment"].value_counts().sort_index()
    for label, count in counts.items():
        percent = count / len(df) * 100
        print(f"{LABEL_MAP[label]}: {count} ({percent:.2f}%)")

def check_cross_split_duplicates(train_df, val_df, test_df):
    print("\n===== CROSS-SPLIT DUPLICATES =====")

    train = set(train_df["sentence"])
    val = set(val_df["sentence"])
    test = set(test_df["sentence"])

    print("Train - Validation:", len(train & val))
    print("Train - Test:", len(train & test))
    print("Validation - Test:", len(val & test))


def main():
    dataset = load_uit_vsfc()

    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()
    test_df = dataset["test"].to_pandas()

    audit_split("train", train_df)
    audit_split("validation", val_df)
    audit_split("test", test_df)

    check_cross_split_duplicates(train_df, val_df, test_df)

    duplicates = train_df[train_df.duplicated(subset="sentence", keep=False)]
    if len(duplicates) > 0:
        print("\n===== TRAIN DUPLICATE DETAILS =====")
        print(duplicates[["sentence", "sentiment", "topic"]].to_string(index=False))

if __name__ == "__main__":
    main()