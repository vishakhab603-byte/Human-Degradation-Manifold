import numpy as np
import pandas as pd

def generate_multimodal_data(n_samples=1000):

    reaction_time = np.random.normal(300, 50, n_samples)
    error_rate = np.random.uniform(0, 0.3, n_samples)
    heart_rate = np.random.normal(75, 10, n_samples)
    sleep_hours = np.random.uniform(4, 9, n_samples)

    label = (
        (reaction_time > 350) |
        (error_rate > 0.2) |
        (sleep_hours < 5)
    ).astype(int)

    df = pd.DataFrame({
        "reaction_time": reaction_time,
        "error_rate": error_rate,
        "heart_rate": heart_rate,
        "sleep_hours": sleep_hours,
        "label": label
    })

    return df
