
import numpy as np


def mc_dropout_predict(model, inputs, n_samples=50):

    preds = [
        model(inputs, training=True)
        for _ in range(n_samples)
    ]

    preds = np.array(preds)

    mean_prediction = preds.mean(axis=0)
    uncertainty = preds.std(axis=0)

    return mean_prediction, uncertainty
