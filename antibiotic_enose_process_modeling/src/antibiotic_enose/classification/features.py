from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def window_summary_features(X):
    X = np.asarray(X, float)
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    last = X[:, -1, :]
    slope = np.diff(X, axis=1).mean(axis=1)
    return np.hstack([mean, std, last, slope])


def pca_tsne_embedding(X, seed=20260913, perplexity=30):
    F = window_summary_features(X)
    Fs = StandardScaler().fit_transform(F)
    pca = PCA(n_components=2, random_state=seed)
    Zp = pca.fit_transform(Fs)
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto",
                max_iter=1000, random_state=seed, init="pca")
    Zt = tsne.fit_transform(Fs)
    return Zp, Zt, pca.explained_variance_ratio_
