from __future__ import annotations
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from .features import window_summary_features


def fit_mahalanobis_reference(X_normal):
    F = window_summary_features(X_normal)
    estimator = EmpiricalCovariance().fit(F)
    d = np.sqrt(estimator.mahalanobis(F))
    threshold = float(d.mean() + 3.0*d.std(ddof=0))
    return estimator, threshold, d


def mahalanobis_scores(estimator, X):
    F = window_summary_features(X)
    return np.sqrt(estimator.mahalanobis(F))
