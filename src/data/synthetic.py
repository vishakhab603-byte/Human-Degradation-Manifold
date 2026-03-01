# src/data/synthetic.py

import numpy as np
import pandas as pd


def generate_synthetic_data(n_samples=1000, random_state=42):

    np.random.seed(random_state)

    # Behavioral signals
    b1 = np.random.normal(0, 1, n_samples)
    b2 = np.random.normal(0, 1, n_samples)

    # Physiological signals
    p1 = np.random.normal(0, 1, n_samples)
    p2 = np.random.normal(0, 1, n_samples)

    # Cross-modal nonlinear degradation function
    interaction_term = 0.7 * (b1 * p1)
    nonlinear_behavior = 0.5 * (b2 ** 2)
    physio_effect = -0.4 * p2

    noise = np.random.normal(0, 0.2, n_samples)

    degradation_score = (
        interaction_term +
        nonlinear_behavior +
        physio_effect +
        noise
    )

    # Sigmoid to bound
    degradation_score = 1 / (1 + np.exp(-degradation_score))

    # Create 3 degradation classes
    labels = np.zeros(n_samples)

    labels[degradation_score >= 0.33] = 1
    labels[degradation_score >= 0.66] = 2

    df = pd.DataFrame({
        "b1": b1,
        "b2": b2,
        "p1": p1,
        "p2": p2,
        "label": labels.astype(int)
    })

    return df
