# src/experiments/train.py

import numpy as np
from tensorflow.keras.utils import to_categorical

from src.config import HDMConfig
from src.data.synthetic import generate_synthetic_data
from src.models.attention_fusion import build_attention_model
from src.models.base_fusion import build_base_model
from src.metrics.hli import compute_hli


def evaluate_models(X, y, config, experiment_name):

    Xb = X[:, 0:2]
    Xp = X[:, 2:4]

    hli_targets = np.argmax(y, axis=1).reshape(-1, 1)

    print(f"\n===== {experiment_name} =====")

    # BASE MODEL
    base_model = build_base_model(config)
    base_history = base_model.fit(
        [Xb, Xp],
        {'classification': y, 'hli': hli_targets},
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=0.2,
        verbose=0
    )
    base_acc = base_history.history['val_classification_accuracy'][-1]

    # ATTENTION MODEL
    attention_model = build_attention_model(config)
    attention_history = attention_model.fit(
        [Xb, Xp],
        {'classification': y, 'hli': hli_targets},
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=0.2,
        verbose=0
    )
    attention_acc = attention_history.history['val_classification_accuracy'][-1]

    print("Base Accuracy:", base_acc)
    print("Attention Accuracy:", attention_acc)

    return base_acc, attention_acc


def run_experiment():

    config = HDMConfig()

    print("Generating synthetic dataset...")
    df = generate_synthetic_data(config.n_samples)

    X = df.drop("label", axis=1).values
    y = to_categorical(df["label"], num_classes=config.num_classes)

    results = {}

    # =====================================================
    # 1️⃣ CLEAN DATA
    # =====================================================
    base_acc, attention_acc = evaluate_models(
        X, y, config, "CLEAN DATA"
    )
    results["clean"] = (base_acc, attention_acc)

    # =====================================================
    # 2️⃣ GAUSSIAN NOISE
    # =====================================================
    noise_level = 0.20
    X_noise = X + np.random.normal(0, noise_level, X.shape)

    base_acc, attention_acc = evaluate_models(
        X_noise, y, config, "GAUSSIAN NOISE"
    )
    results["noise"] = (base_acc, attention_acc)

    # =====================================================
    # 3️⃣ MODALITY CORRUPTION
    # Simulate behavioral sensor failure
    # =====================================================
    X_corrupt = X.copy()

    corruption_ratio = 0.3
    n_corrupt = int(len(X_corrupt) * corruption_ratio)

    indices = np.random.choice(len(X_corrupt), n_corrupt, replace=False)

    # Corrupt only behavioral features (first 2 columns)
    X_corrupt[indices, 0:2] = np.random.normal(
        0, 2.5, (n_corrupt, 2)
    )

    base_acc, attention_acc = evaluate_models(
        X_corrupt, y, config, "MODALITY CORRUPTION (Behavioral)"
    )
    results["corruption"] = (base_acc, attention_acc)

    # =====================================================
    # HLI Computation
    # =====================================================
    hli_values, tau = compute_hli(X)

    print("\n===== FINAL SUMMARY =====")
    print("Clean        -> Base:", results["clean"][0],
          "Attention:", results["clean"][1])
    print("Noise        -> Base:", results["noise"][0],
          "Attention:", results["noise"][1])
    print("Corruption   -> Base:", results["corruption"][0],
          "Attention:", results["corruption"][1])
    print("HLI Threshold Tau:", tau)

    return results, tau
