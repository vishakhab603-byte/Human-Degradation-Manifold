# src/train.py

from data_generator import generate_multimodal_data
from preprocessing import preprocess_data
from models import build_fusion_model
from hli import compute_hli
import numpy as np


def train_pipeline():

    print("Generating Data...")
    df = generate_multimodal_data()

    print("Preprocessing Data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Split modalities
    Xb_train = X_train[:, 0:2]
    Xp_train = X_train[:, 2:4]

    Xb_test = X_test[:, 0:2]
    Xp_test = X_test[:, 2:4]

    print("Building Model...")
    model = build_fusion_model()

    # Create HLI target (normalized degradation severity)
    hli_train = y_train.argmax(axis=1) / 2.0
    hli_test = y_test.argmax(axis=1) / 2.0

    print("Training Model...")
    model.fit(
        [Xb_train, Xp_train],
        {
            "classification": y_train,
            "hli": hli_train
        },
        epochs=20,
        batch_size=32,
        validation_split=0.2
    )

    print("Evaluating Model...")
    results = model.evaluate(
        [Xb_test, Xp_test],
        {
            "classification": y_test,
            "hli": hli_test
        }
    )

    print("Evaluation Results:", results)

    # Compute HLI on full dataset
    print("Computing Human Limit Index...")
    hli_values, tau = compute_hli(df.drop("label", axis=1).values)

    print("HLI Threshold (tau):", tau)

    return model, hli_values


if __name__ == "__main__":
    train_pipeline()
