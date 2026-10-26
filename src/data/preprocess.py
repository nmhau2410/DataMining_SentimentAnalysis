from pathlib import Path
import re
import unicodedata
import py_vncorenlp

from src.data.load_data import load_uit_vsfc


ROOT_DIR = Path(__file__).resolve().parents[2]
VNCORENLP_DIR = ROOT_DIR / "tools" / "vncorenlp"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"

def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_conflicting_labels(df):
    label_counts = df.groupby("sentence")["sentiment"].nunique()
    conflicts = label_counts[label_counts > 1].index

    print("Conflicting sentences:", len(conflicts))

    df = df[~df["sentence"].isin(conflicts)].copy()

    return df.reset_index(drop=True)

def load_segmenter():
    jar_file = VNCORENLP_DIR / "VnCoreNLP-1.2.jar"

    if not jar_file.exists():
        raise FileNotFoundError( f"Không tìm thấy: {jar_file}")

    return py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(VNCORENLP_DIR))

def segment_text(text, segmenter):
    result = segmenter.word_segment(text)
    return " ".join(result)


def preprocess(df, segmenter):
    df = df[["sentence", "sentiment"]].copy()
    df["text_normalized"] = df["sentence"].apply(normalize_text)
    df["text_segmented"] = df["text_normalized"].apply(lambda text: segment_text(text, segmenter))
    return df


def main():
    dts = load_uit_vsfc()

    train_df = dts["train"].to_pandas()
    val_df = dts["validation"].to_pandas()
    test_df = dts["test"].to_pandas()

    print("Train before cleaning:", len(train_df))
    train_df = remove_conflicting_labels(train_df)
    print("Train after cleaning:", len(train_df))

    segmenter = load_segmenter()
    train_df = preprocess(train_df, segmenter)
    val_df = preprocess(val_df, segmenter)
    test_df = preprocess(test_df, segmenter)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(OUTPUT_DIR / "validation.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False, encoding="utf-8-sig")

    print("\nDONE:")
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    print("\nExample:")
    print("Raw       :", train_df.iloc[0]["sentence"])
    print("Normalized:", train_df.iloc[0]["text_normalized"])
    print("Segmented :", train_df.iloc[0]["text_segmented"])

if __name__ == "__main__":
    main()