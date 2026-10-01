from __future__ import annotations
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from ..constants import SENSOR_COLS
from ..preprocessing import relative_progress


def source_coordinates(X, source_repr, n_components=2):
    latent=source_repr.transform(np.asarray(X,float))
    latent=StandardScaler().fit_transform(latent)
    n_components=min(n_components,latent.shape[1],len(latent))
    return PCA(n_components=n_components).fit_transform(latent)


def selector_embedding(df, source_repr):
    X=df[SENSOR_COLS].to_numpy(float)
    pcs=source_coordinates(X,source_repr,2)
    return np.c_[relative_progress(df["Time_h"].to_numpy(float)),pcs[:,0]]


def kennard_stone(F,k):
    F=np.asarray(F,float)
    if k>=len(F): return np.arange(len(F),dtype=int)
    Z=(F-F.mean(0))/(F.std(0)+1e-12)
    dm=np.sqrt(((Z[:,None,:]-Z[None,:,:])**2).sum(2))
    i,j=np.unravel_index(np.argmax(dm),dm.shape)
    selected=[int(i)]
    if j!=i: selected.append(int(j))
    while len(selected)<k:
        rem=np.array([r for r in range(len(Z)) if r not in selected],int)
        md=np.min(dm[rem][:,selected],axis=1)
        selected.append(int(rem[np.argmax(md)]))
    return np.array(selected[:k],int)


def stratified_outer_split(df,rng,test_fraction=0.25):
    t=df["Time_h"].to_numpy(float);prog=relative_progress(t)
    strata=np.minimum((prog*3).astype(int),2)
    n=len(df);n_test=max(3,int(round(n*test_fraction)))
    test=[]
    for s in range(3):
        ids=np.where(strata==s)[0]
        if len(ids): test.append(int(rng.choice(ids)))
    remain=[i for i in range(n) if i not in test]
    if n_test>len(test): test.extend(map(int,rng.choice(remain,size=n_test-len(test),replace=False)))
    test=np.array(sorted(set(test)),int)
    pool=np.array([i for i in range(n) if i not in set(test)],int)
    return pool,test


def dynamic_prediction(t,y,train_idx,test_idx):
    t=np.asarray(t,float);y=np.asarray(y,float);train_idx=np.asarray(train_idx,int);test_idx=np.asarray(test_idx,int)
    order=np.argsort(t[train_idx])
    f=interp1d(t[train_idx][order],y[train_idx][order],kind="linear",
               bounds_error=False,fill_value="extrapolate")
    return np.asarray(f(t[test_idx]),float)


def source_guided_prediction(df,y,train_idx,test_idx,source_repr,ridge_alpha=10.0,source_shrinkage=0.10):
    t=df["Time_h"].to_numpy(float)
    y=np.asarray(y,float);train_idx=np.asarray(train_idx,int);test_idx=np.asarray(test_idx,int)
    base=dynamic_prediction(t,y,train_idx,test_idx)
    if len(train_idx)<4: return base
    pcs=source_coordinates(df[SENSOR_COLS].to_numpy(float),source_repr,2)
    residuals=[];z=[]
    for idx in train_idx:
        sub=train_idx[train_idx!=idx]
        loo=dynamic_prediction(t,y,sub,np.array([idx]))[0]
        residuals.append(y[idx]-loo);z.append(pcs[idx])
    z=np.asarray(z,float);residuals=np.asarray(residuals,float)
    sc=StandardScaler().fit(z)
    model=Ridge(alpha=ridge_alpha).fit(sc.transform(z),residuals)
    corr=model.predict(sc.transform(pcs[test_idx]))
    return base+source_shrinkage*corr
