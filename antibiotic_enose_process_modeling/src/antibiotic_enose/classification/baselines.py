from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from .features import window_summary_features
from ..metrics import classification_metrics


def benchmark_classifiers(X, y, test_size=0.30, seed=20260913):
    """Sample-level 7:3 benchmark used for the manuscript classification analysis."""
    F = window_summary_features(X)
    Xtr, Xte, ytr, yte = train_test_split(
        F, y, test_size=test_size, stratify=y, random_state=seed
    )
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    models = {
        "RF": RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1),
        "SVM_RBF": SVC(C=10, gamma="scale", kernel="rbf", random_state=seed),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=seed),
    }
    rows = []
    fitted = {}
    for name, model in models.items():
        xxtr, xxte = (Xtr, Xte) if name == "RF" else (Xtr_s, Xte_s)
        model.fit(xxtr, ytr)
        pred = model.predict(xxte)
        rows.append({"model": name, **classification_metrics(yte, pred)})
        fitted[name] = model
    return pd.DataFrame(rows), fitted, (Xte, yte)
