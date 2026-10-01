from __future__ import annotations
import numpy as np
import pandas as pd
from ..metrics import regression_metrics
from .fewshot import stratified_outer_split, selector_embedding, kennard_stone, source_guided_prediction


def learning_curve_for_variable(product,df,variable,source_repr,fractions=(.2,.4,.6,.8,1.0),
                                n_repeats=100,outer_test_fraction=.25,seed=20260913,
                                ridge_alpha=10.0,source_shrinkage=.10):
    y=df[variable].to_numpy(float);Z=selector_embedding(df,source_repr)
    rng=np.random.default_rng(seed);rows=[]
    for rep in range(n_repeats):
        pool,test=stratified_outer_split(df,rng,outer_test_fraction)
        for frac in fractions:
            nlab=max(2,int(np.ceil(len(pool)*frac)));nlab=min(nlab,len(pool))
            chosen=pool if frac>=1 else pool[kennard_stone(Z[pool],nlab)]
            pred=source_guided_prediction(df,y,chosen,test,source_repr,ridge_alpha,source_shrinkage)
            rows.append({"Product":product,"Variable":variable,"Repeat":rep,"Fraction":float(frac),
                         "N_labels":int(len(chosen)),**regression_metrics(y[test],pred)})
    return pd.DataFrame(rows)


def summarize_learning_curves(repeat_level):
    g=repeat_level.groupby(["Product","Variable","Fraction"],as_index=False)
    out=g.agg(Median_N_labels=("N_labels","median"),Median_R2=("R2","median"),
              R2_Q25=("R2",lambda x:np.quantile(x,.25)),R2_Q75=("R2",lambda x:np.quantile(x,.75)),
              Median_RMSE=("RMSE","median"),Median_NRMSE=("NRMSE","median"))
    out["Calibration_pool_fraction_pct"]=(out["Fraction"]*100).round().astype(int)
    return out


def threshold_table(summary,threshold=.80):
    rows=[]
    for (prod,var),g in summary.groupby(["Product","Variable"],sort=False):
        g=g.sort_values("Fraction");ok=g[g["Median_R2"]>=threshold]
        if len(ok):
            r=ok.iloc[0];status=int(r["Calibration_pool_fraction_pct"])
        else:
            r=g.loc[g["Median_R2"].idxmax()];status="Not reached"
        rows.append({"Product":prod,"Variable":var,"Minimum_fraction_for_R2_ge_threshold":status,
                     "N_labels":int(round(r["Median_N_labels"])),"R2_at_threshold_or_best":float(r["Median_R2"]),
                     "RMSE_at_threshold_or_best":float(r["Median_RMSE"])})
    return pd.DataFrame(rows)
