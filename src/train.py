# src/train.py

import numpy as np
from src.data_generator import generate_multimodal_data
from src.preprocessing import preprocess_data
from src.models import build_fusion_model
from src.hli import compute_hli


def train_pipeline(n_samples=1000, epochs=20):

    print("Generating dataset...")
    df = generate_multimodal_data(n_samples)

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Split modalities
    Xb_train = X_train[:, 0:2]
    Xp_train = X_train[:, 2:4]

    Xb_test = X_test[:, 0:2]
    Xp_test = X_test[:, 2:4]

    print("Building model...")
    model = build_fusion_model()

    # HLI targets (severity 0,1,2)
    hli_train = np.argmax(y_train, axis=1).reshape(-1, 1)
    hli_test = np.argmax(y_test, axis=1).reshape(-1, 1)

    print("Training model...")
    history = model.fit(
        [Xb_train, Xp_train],
        {
            "classification": y_train,
            "hli": hli_train
        },
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    print("Evaluating model...")
    results = model.evaluate(
        [Xb_test, Xp_test],
        {
            "classification": y_test,
            "hli": hli_test
        },
        verbose=0
    )

    print("Computing HLI trajectory...")
    X_full = df.drop("label", axis=1).values
    hli_values, tau = compute_hli(X_full)

    return {
        "model": model,
        "history": history,
        "results": results,
        "hli_values": hli_values,
        "tau": tau,
        "df": df
    }
