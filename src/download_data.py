"""
Pulls FER-2013 down from the HF mirror and lays it out as
data/train/<class>, data/test/<class> so it matches the folder-per-class
convention the rest of the pipeline expects.

The original challenge ships 3 splits: train, publicTest, privateTest. I
fold publicTest into train (more data to carve my own stratified val split
from later, same as I did in the tabular project) and keep privateTest as
the sealed test set, never touched until final evaluation.

Only needs to be run once. Kaggle also hosts this dataset (msambare/fer2013)
in the same folder layout if the HF mirror ever goes down.
"""
import os

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO_ID = "Aaryan333/fer2013_train_publicTest_privateTest"
LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

SPLIT_FILES = {
    "train": "data/train-00000-of-00001-5eab84e1c6a2fc27.parquet",
    "publicTest": "data/publicTest-00000-of-00001-f41bb7384b8aad6e.parquet",
    "privateTest": "data/privateTest-00000-of-00001-4b8a0715cf1b7560.parquet",
}

# both train and publicTest land in data/train, privateTest becomes data/test
SPLIT_TO_DIR = {
    "train": "train",
    "publicTest": "train",
    "privateTest": "test",
}

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


def materialize_split(split_name, out_dir):
    local_path = hf_hub_download(repo_id=REPO_ID, filename=SPLIT_FILES[split_name], repo_type="dataset")
    table = pq.read_table(local_path)
    labels = table.column("label").to_pylist()
    images = table.column("image").to_pylist()

    counts = {name: 0 for name in LABELS}
    for label_idx, img in zip(labels, images):
        class_name = LABELS[label_idx]
        class_dir = os.path.join(out_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        counts[class_name] += 1
        fname = f"{split_name}_{counts[class_name]:05d}.png"
        with open(os.path.join(class_dir, fname), "wb") as f:
            f.write(img["bytes"])
    return counts


if __name__ == "__main__":
    for split_name, dir_name in SPLIT_TO_DIR.items():
        out_dir = os.path.join(DATA_ROOT, dir_name)
        print(f"writing {split_name} -> data/{dir_name}/ ...")
        counts = materialize_split(split_name, out_dir)
        print(counts)
