# src/hli.py

import numpy as np
from scipy.spatial.distance import mahalanobis

def compute_hli(X, baseline_end=400):
    """
    Compute Human Limit Index (HLI) using Mahalanobis distance.
    Baseline is assumed to be the stable initial region.
    """

    # Compute baseline mean and covariance
    mean_vec = np.mean(X[:baseline_end], axis=0)
    cov_matrix = np.cov(X[:baseline_end].T)

    # Add small regularization for numerical stability
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6

    inv_cov = np.linalg.inv(cov_matrix)

    # Compute Mahalanobis distance for each point
    distances = [
        mahalanobis(x, mean_vec, inv_cov) for x in X
    ]

    distances = np.array(distances)

    # Define threshold tau from baseline
    tau = np.mean(distances[:baseline_end]) + 2 * np.std(distances[:baseline_end])

    # Normalize to get HLI
    hli = distances / tau

    return hli, tau
