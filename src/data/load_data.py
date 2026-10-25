from datasets import load_dataset

DATASET_URL = (
    "https://huggingface.co/datasets/"
    "uitnlp/vietnamese_students_feedback/"
    "resolve/refs%2Fconvert%2Fparquet/default"
)

DATA_FILES = {
    "train": f"{DATASET_URL}/train/0000.parquet",
    "validation": f"{DATASET_URL}/validation/0000.parquet",
    "test": f"{DATASET_URL}/test/0000.parquet",
}


def load_uit_vsfc():
    dts = load_dataset("parquet", data_files=DATA_FILES)
    print(dts)
    return dts


if __name__ == "__main__":
    dts = load_uit_vsfc()
