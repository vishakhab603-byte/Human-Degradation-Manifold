import numpy as np
import pandas as pd

def generate_multimodal_data(n_samples=1000):

    # Behavioral signals
    reaction_time = np.random.normal(300, 40, n_samples)
    error_rate = np.random.uniform(0, 0.3, n_samples)

    # Physiological signals
    heart_rate = np.random.normal(75, 8, n_samples)
    sleep_hours = np.random.uniform(4, 9, n_samples)

    label = []

    for rt, err, hr, sleep in zip(reaction_time, error_rate, heart_rate, sleep_hours):

        # Severe degradation
        if rt > 360 or err > 0.25 or sleep < 4.8:
            label.append(2)

        # Moderate degradation
        elif rt > 320 or err > 0.15 or sleep < 6:
            label.append(1)

        # Stable
        else:
            label.append(0)

    df = pd.DataFrame({
        "reaction_time": reaction_time,
        "error_rate": error_rate,
        "heart_rate": heart_rate,
        "sleep_hours": sleep_hours,
        "label": label
    })

    return df
