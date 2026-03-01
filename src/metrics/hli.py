import numpy as np
from scipy.spatial.distance import mahalanobis


def compute_hli(X, baseline_end=400):

    mean_vec = np.mean(X[:baseline_end], axis=0)
    cov_matrix = np.cov(X[:baseline_end].T)
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
    inv_cov = np.linalg.inv(cov_matrix)

    distances = [
        mahalanobis(x, mean_vec, inv_cov)
        for x in X
    ]

    distances = np.array(distances)

    tau = (
        np.mean(distances[:baseline_end]) +
        2 * np.std(distances[:baseline_end])
    )

    return distances, tau
