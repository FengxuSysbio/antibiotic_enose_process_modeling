"""Unstructured fermentation-kinetic utilities.

The model family follows the manuscript Methods. Exact numerical solver and parameter
bounds used for the final publication should be matched to the final archived analysis.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


def monod_mu(S, mumax, Ks):
    return mumax*S/(Ks+S+1e-12)


def haldane_mu(S, mumax, Ks, Ki):
    return mumax*S/(Ks+S+S*S/(Ki+1e-12)+1e-12)


def logistic_growth(X, mu, Xmax):
    return mu*X*(1-X/(Xmax+1e-12))


def luedeking_piret(dXdt, X, alpha, beta):
    return alpha*dXdt+beta*X


def rhs_model7(t, z, p):
    """Integrated reconstruction used in the archived manuscript workflow.

    z = [PMV, RS, TS, NH4, Titer]
    p = [mumax, Ks, Ki, Xmax, Yxs, Yxn, alpha, beta, k_ts, k_rs, k_n]
    """
    X,RS,TS,N,P=np.maximum(z,0.0)
    mumax,Ks,Ki,Xmax,Yxs,Yxn,alpha,beta,k_ts,k_rs,k_n=p
    mu=haldane_mu(RS,mumax,Ks,Ki)
    dX=logistic_growth(X,mu,Xmax)
    dRS=-(1/max(Yxs,1e-8))*dX-k_rs*X
    dTS=-k_ts*max(RS,0.0)*X
    dN=-(1/max(Yxn,1e-8))*dX-k_n*X
    dP=luedeking_piret(dX,X,alpha,beta)
    return [dX,dRS,dTS,dN,dP]


def simulate_model7(t_eval,z0,p,method="LSODA"):
    t_eval=np.asarray(t_eval,float)
    sol=solve_ivp(lambda t,z: rhs_model7(t,z,p),(float(t_eval.min()),float(t_eval.max())),
                  np.asarray(z0,float),t_eval=t_eval,method=method,rtol=1e-7,atol=1e-9)
    if not sol.success: raise RuntimeError(sol.message)
    return sol.y.T


def fit_model7(t,Y,max_nfev=5000):
    t=np.asarray(t,float);Y=np.asarray(Y,float);z0=np.maximum(Y[0],1e-8)
    p0=np.array([0.023,30.0,23.0,max(Y[:,0].max()*1.2,1.0),1.5,2.0,0.1,0.07,1e-3,1e-3,1e-3])
    lb=np.array([1e-5,1e-5,1e-5,max(Y[:,0].max(),1e-3),1e-3,1e-3,0,0,0,0,0])
    ub=np.array([1.0,500,500,max(Y[:,0].max()*10,10),100,100,10,10,10,10,10])
    scale=np.nanstd(Y,axis=0);scale[scale<1e-8]=1
    def residual(p): return ((simulate_model7(t,z0,p)-Y)/scale).ravel()
    res=least_squares(residual,p0,bounds=(lb,ub),max_nfev=max_nfev)
    return res.x,res.cost,simulate_model7(t,z0,res.x)


def kinetic_augmentation(t_real,Y_real,p_hat,n_points=1000,target_names=("PMV","RS","TS","NH4","Titer")):
    t_dense=np.linspace(float(np.min(t_real)),float(np.max(t_real)),int(n_points))
    z0=np.maximum(np.asarray(Y_real,float)[0],1e-8)
    Y_dense=simulate_model7(t_dense,z0,p_hat)
    out=pd.DataFrame(Y_dense,columns=list(target_names))
    out.insert(0,"Time_h",t_dense)
    out["Data_origin"]="kinetic_model_generated"
    return out
