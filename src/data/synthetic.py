# src/data/synthetic.py

import numpy as np
import pandas as pd


def generate_synthetic_data(n_samples=1200):
    """
    Generates a clean, separable 3-class multimodal dataset.
    
    Classes:
    0 → Stable
    1 → Moderate Degradation
    2 → Severe Degradation
    """

    np.random.seed(42)

    samples_per_class = n_samples // 3

    # ==============================
    # CLASS 0 — STABLE
    # ==============================
    behavior_0 = np.random.normal(
        loc=[0.2, 0.3],
        scale=0.04,
        size=(samples_per_class, 2)
    )

    physio_0 = np.random.normal(
        loc=[0.3, 0.2],
        scale=0.04,
        size=(samples_per_class, 2)
    )

    # ==============================
    # CLASS 1 — MODERATE
    # ==============================
    behavior_1 = np.random.normal(
        loc=[0.6, 0.55],
        scale=0.04,
        size=(samples_per_class, 2)
    )

    physio_1 = np.random.normal(
        loc=[0.55, 0.6],
        scale=0.04,
        size=(samples_per_class, 2)
    )

    # ==============================
    # CLASS 2 — SEVERE
    # ==============================
    behavior_2 = np.random.normal(
        loc=[0.9, 0.85],
        scale=0.04,
        size=(samples_per_class, 2)
    )

    physio_2 = np.random.normal(
        loc=[0.85, 0.9],
        scale=0.04,
        size=(samples_per_class, 2)
    )

    # Stack modalities
    behavior = np.vstack([behavior_0, behavior_1, behavior_2])
    physio = np.vstack([physio_0, physio_1, physio_2])

    labels = np.array(
        [0] * samples_per_class +
        [1] * samples_per_class +
        [2] * samples_per_class
    )

    # Shuffle
    indices = np.random.permutation(len(labels))
    behavior = behavior[indices]
    physio = physio[indices]
    labels = labels[indices]

    df = pd.DataFrame({
        "b1": behavior[:, 0],
        "b2": behavior[:, 1],
        "p1": physio[:, 0],
        "p2": physio[:, 1],
        "label": labels
    })

    return df
