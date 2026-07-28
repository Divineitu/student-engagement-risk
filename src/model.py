"""CNN architectures used in the notebook's experiment ladder (E1/E2/E3)."""
from tensorflow.keras import layers, models, regularizers

INPUT_SHAPE = (48, 48, 1)
NUM_CLASSES = 7


def build_baseline_cnn():
    """E1 - plain conv stack, no regularization, just to see where we start from."""
    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_regularized_cnn(l2=1e-4):
    """E2/E3 - dropout + light L2 + early stopping, used for both the
    class-weight experiment and the deployed model (retrain.py fine-tunes
    this one further). Dropped BatchNorm after it kept crashing the CPU
    build during training on my machine - L2 + dropout does the job here
    without it."""
    reg = regularizers.l2(l2)
    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE),

        layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=reg),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=reg),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=reg),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu", kernel_regularizer=reg),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
