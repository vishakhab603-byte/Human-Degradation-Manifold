# src/experiments/train.py

import numpy as np
from tensorflow.keras.utils import to_categorical

from src.config import HDMConfig
from src.data.synthetic import generate_synthetic_data
from src.models.attention_fusion import build_attention_model
from src.models.base_fusion import build_base_model
from src.metrics.hli import compute_hli


def run_experiment():

    config = HDMConfig()

    print("Generating synthetic dataset...")
    df = generate_synthetic_data(config.n_samples)

    X = df.drop("label", axis=1).values
    y = to_categorical(
        df["label"],
        num_classes=config.num_classes
    )

    # Add Gaussian noise
    noise_level = 0.15
    X_noisy = X + np.random.normal(0, noise_level, X.shape)

    Xb = X_noisy[:, 0:2]
    Xp = X_noisy[:, 2:4]

    hli_targets = np.argmax(y, axis=1).reshape(-1, 1)

    # ==============================
    # BASE MODEL
    # ==============================
    print("\nTraining Base Fusion Model (Noisy Data)...")
    base_model = build_base_model(config)

    base_history = base_model.fit(
        [Xb, Xp],
        {'classification': y, 'hli': hli_targets},
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=0.2,
        verbose=0
    )

    base_val_acc = base_history.history['val_classification_accuracy'][-1]

    # ==============================
    # ATTENTION MODEL
    # ==============================
    print("\nTraining Attention Fusion Model (Noisy Data)...")
    attention_model = build_attention_model(config)

    attention_history = attention_model.fit(
        [Xb, Xp],
        {'classification': y, 'hli': hli_targets},
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=0.2,
        verbose=0
    )

    attention_val_acc = attention_history.history['val_classification_accuracy'][-1]

    # ==============================
    # HLI COMPUTATION
    # ==============================
    hli_values, tau = compute_hli(X_noisy)

    print("\n===== NOISE EXPERIMENT RESULTS =====")
    print("Base Fusion Validation Accuracy:", base_val_acc)
    print("Attention Fusion Validation Accuracy:", attention_val_acc)
    print("HLI Threshold Tau:", tau)

    return {
        "base_val_acc": base_val_acc,
        "attention_val_acc": attention_val_acc,
        "tau": tau
    }
