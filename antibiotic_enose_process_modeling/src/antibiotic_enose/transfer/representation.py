from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from . import __init__  # noqa: F401
from ..constants import SENSOR_COLS, GENTAMICIN_TARGETS


class GentamicinSourceRepresentation:
    def __init__(self,n_components=6):
        self.n_components=n_components

    def fit(self,source_df):
        X=source_df[SENSOR_COLS].to_numpy(float)
        Y=source_df[GENTAMICIN_TARGETS].to_numpy(float)
        self.x_scaler=StandardScaler().fit(X)
        self.y_scaler=StandardScaler().fit(Y)
        Xs=self.x_scaler.transform(X);Ys=self.y_scaler.transform(Y)
        ncomp=min(self.n_components,Xs.shape[1],len(Xs)-1)
        self.pls=PLSRegression(n_components=ncomp,scale=False).fit(Xs,Ys)
        return self

    def transform(self,X):
        X=np.asarray(X,float)
        return self.pls.transform(self.x_scaler.transform(X))
