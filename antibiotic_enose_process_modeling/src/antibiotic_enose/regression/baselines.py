from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from ..constants import SENSOR_COLS
from ..metrics import regression_metrics


def benchmark_regressors(train_df, test_df, target, seed=20260913):
    Xtr = train_df[SENSOR_COLS].to_numpy(float)
    Xte = test_df[SENSOR_COLS].to_numpy(float)
    ytr = train_df[target].to_numpy(float)
    yte = test_df[target].to_numpy(float)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    models = {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=1e-3, max_iter=10000),
        "ElasticNet": ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=10000),
        "RandomForest": RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1),
        "GBDT": GradientBoostingRegressor(random_state=seed),
        "SVR": SVR(C=10, epsilon=0.05, gamma="scale"),
        "KNNR": KNeighborsRegressor(n_neighbors=5),
    }
    rows=[]; fitted={}
    for name,m in models.items():
        raw = name in {"RandomForest","GBDT"}
        m.fit(Xtr if raw else Xtr_s, ytr)
        pred=m.predict(Xte if raw else Xte_s)
        rows.append({"Target":target,"Model":name,**regression_metrics(yte,pred)})
        fitted[name]=m
    return pd.DataFrame(rows), fitted
