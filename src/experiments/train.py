# src/experiments/train.py

import numpy as np
from tensorflow.keras.utils import to_categorical

from src.config import HDMConfig
from src.data.synthetic import generate_synthetic_data
from src.models.attention_fusion import build_attention_model
from src.models.base_fusion import build_base_model
from src.metrics.hli import compute_hli


def evaluate_models(X, y, config):

    Xb = X[:, 0:2]
    Xp = X[:, 2:4]

    hli_targets = np.argmax(y, axis=1).reshape(-1, 1)

    # Base model
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

    # Attention model
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

    return base_acc, attention_acc


def run_experiment():

    config = HDMConfig()

    seeds = list(range(10))

    clean_results = []
    noise_results = []
    corruption_results = []

    for seed in seeds:

        np.random.seed(seed)

        df = generate_synthetic_data(config.n_samples, random_state=seed)

        X = df.drop("label", axis=1).values
        y = to_categorical(df["label"], num_classes=config.num_classes)

        # CLEAN
        base_acc, attention_acc = evaluate_models(X, y, config)
        clean_results.append((base_acc, attention_acc))

        # NOISE
        X_noise = X + np.random.normal(0, 0.20, X.shape)
        base_acc, attention_acc = evaluate_models(X_noise, y, config)
        noise_results.append((base_acc, attention_acc))

        # MODALITY CORRUPTION
        X_corrupt = X.copy()
        n_corrupt = int(len(X_corrupt) * 0.3)
        indices = np.random.choice(len(X_corrupt), n_corrupt, replace=False)
        X_corrupt[indices, 0:2] = np.random.normal(0, 2.5, (n_corrupt, 2))

        base_acc, attention_acc = evaluate_models(X_corrupt, y, config)
        corruption_results.append((base_acc, attention_acc))

    def summarize(results):
        base = np.array([r[0] for r in results])
        attn = np.array([r[1] for r in results])
        return (
            base.mean(), base.std(),
            attn.mean(), attn.std()
        )

    clean_summary = summarize(clean_results)
    noise_summary = summarize(noise_results)
    corruption_summary = summarize(corruption_results)

    print("\n===== MULTI-RUN STATISTICAL RESULTS =====")

    print("\nCLEAN")
    print("Base      -> Mean:", clean_summary[0], "Std:", clean_summary[1])
    print("Attention -> Mean:", clean_summary[2], "Std:", clean_summary[3])

    print("\nNOISE")
    print("Base      -> Mean:", noise_summary[0], "Std:", noise_summary[1])
    print("Attention -> Mean:", noise_summary[2], "Std:", noise_summary[3])

    print("\nCORRUPTION")
    print("Base      -> Mean:", corruption_summary[0], "Std:", corruption_summary[1])
    print("Attention -> Mean:", corruption_summary[2], "Std:", corruption_summary[3])

    return {
        "clean": clean_summary,
        "noise": noise_summary,
        "corruption": corruption_summary
    }
