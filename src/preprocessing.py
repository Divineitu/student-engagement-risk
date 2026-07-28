"""
Everything here gets used three places: the training notebook, the retrain
job, and the API's single-image prediction path. Keeping it in one file
means the model always sees inputs preprocessed the same way, whether that
image came from the training folders or a user's upload.
"""
import os

import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = (48, 48)
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def list_files_labels(directory):
    """walks data/<split>/<class>/*.png and returns matching filepath + label lists"""
    filepaths, labels = [], []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            filepaths.append(os.path.join(class_dir, fname))
            labels.append(class_idx)
    return filepaths, labels


def _decode_and_resize(filepath, label):
    raw = tf.io.read_file(filepath)
    img = tf.image.decode_png(raw, channels=1)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img, label


def make_dataset_from_files(filepaths, labels, batch_size=64, shuffle=True, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=seed)
    ds = ds.map(_decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_pipeline(directory, batch_size=64, shuffle=False):
    filepaths, labels = list_files_labels(directory)
    return make_dataset_from_files(filepaths, labels, batch_size=batch_size, shuffle=shuffle)


def preprocess_single(image):
    """image is a PIL.Image opened from an upload, any mode/size -> (1,48,48,1) float32 in [0,1]"""
    img = image.convert("L").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)


def preprocess_single_from_path(path):
    with Image.open(path) as img:
        return preprocess_single(img)
