# src/data_generator.py

import numpy as np
import pandas as pd
def load_real_dataset(path):
    import pandas as pd
    df = pd.read_csv(path)
    return df
    def generate_hybrid_dataset(real_df):
    synthetic_df = generate_multimodal_data(n_samples=len(real_df))
    
    # Replace behavioral columns with real reaction time
    synthetic_df["reaction_time"] = real_df["reaction_time"]
    
    return synthetic_df
def generate_multimodal_data(n=1500, seed=42):
    np.random.seed(seed)

    reaction_time = []
    error_rate = []
    heart_rate = []
    sleep_hours = []
    labels = []

    for t in range(n):

        # Stable Phase
        if t < 500:
            rt = np.random.normal(300, 8)
            err = np.random.binomial(1, 0.02)
            hr = np.random.normal(70, 3)
            sleep = np.random.normal(7.5, 0.3)
            label = 0

        # Gradual Degradation
        elif t < 1100:
            rt = np.random.normal(300 + (t-500)*0.05, 15)
            err = np.random.binomial(1, 0.02 + (t-500)*0.00015)
            hr = np.random.normal(70 + (t-500)*0.02, 5)
            sleep = np.random.normal(7 - (t-500)*0.001, 0.4)
            label = 1

        # Breakdown Phase
        else:
            rt = np.random.normal(380, 40)
            err = np.random.binomial(1, 0.25)
            hr = np.random.normal(85, 10)
            sleep = np.random.normal(5.5, 0.5)
            label = 2

        reaction_time.append(rt)
        error_rate.append(err)
        heart_rate.append(hr)
        sleep_hours.append(sleep)
        labels.append(label)

    df = pd.DataFrame({
        "reaction_time": reaction_time,
        "error_rate": error_rate,
        "heart_rate": heart_rate,
        "sleep_hours": sleep_hours,
        "label": labels
    })

    return df
