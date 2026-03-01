
import numpy as np
from tensorflow.keras.utils import to_categorical

from src.config import HDMConfig
from src.data.synthetic import generate_synthetic_data
from src.models.attention_fusion import build_attention_model
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

    Xb = X[:, 0:2]
    Xp = X[:, 2:4]

    print("Building attention fusion model...")
    model = build_attention_model(config)

    hli_targets = np.argmax(y, axis=1).reshape(-1, 1)

    print("Training...")
    history = model.fit(
        [Xb, Xp],
        {
            'classification': y,
            'hli': hli_targets
        },
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=0.2
    )

    print("Computing HLI...")
    hli_values, tau = compute_hli(X)

    return model, history, tau
