"""ml/tensorflow/train.py — minimal tf.keras training loop skeleton.

TensorBoard + WandB callback. Fill in dataset / model for real task.
"""

from __future__ import annotations

import os
from pathlib import Path

import tensorflow as tf

try:
    import wandb
    from wandb.keras import WandbMetricsLogger
except ImportError:
    wandb = None  # type: ignore
    WandbMetricsLogger = None  # type: ignore


def build_model():
    """TODO: replace with real architecture (keras_cv / keras_nlp)."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(16,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )


def build_dataset(batch_size: int):
    """TODO: real dataset. Placeholder = random."""
    x = tf.random.normal((128, 16))
    y = tf.random.uniform((128,), 0, 2, dtype=tf.int32)
    return tf.data.Dataset.from_tensor_slices((x, y)).shuffle(128).batch(batch_size)


def main():
    batch_size = int(os.getenv("BATCH_SIZE", "8"))
    epochs = int(os.getenv("EPOCHS", "1"))
    lr = float(os.getenv("LR", "1e-3"))

    # Mixed precision (Ampere+: bf16; older: fp16)
    mixed = os.getenv("MIXED_PRECISION", "fp16")
    if mixed in {"fp16", "mixed_float16"}:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    elif mixed in {"bf16", "mixed_bfloat16"}:
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    ds = build_dataset(batch_size)

    log_dir = Path("experiments/tf-logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=str(log_dir))]

    if wandb and WandbMetricsLogger and os.getenv("WANDB_API_KEY"):
        wandb.init(project="hacknation-2026", config={"lr": lr, "bs": batch_size, "epochs": epochs})
        callbacks.append(WandbMetricsLogger())

    model.fit(ds, epochs=epochs, callbacks=callbacks)

    ckpt_dir = Path("experiments/tf-ckpts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save(ckpt_dir / "last.keras")

    if wandb and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
