#!/usr/bin/env python
from pathlib import Path
import argparse
import pandas as pd
from antibiotic_enose.io import load_paired_regression_batches
from antibiotic_enose.constants import GENTAMICIN_TARGETS
from antibiotic_enose.regression.baselines import benchmark_regressors


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',required=True);ap.add_argument('--external-batch',required=True)
    ap.add_argument('--output-dir',required=True);ap.add_argument('--seed',type=int,default=20260913)
    args=ap.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    df=load_paired_regression_batches(args.data_dir)
    train=df[df.Batch!=args.external_batch].copy();test=df[df.Batch==args.external_batch].copy()
    if test.empty: raise ValueError(f'External batch {args.external_batch!r} not found.')
    rows=[]
    for target in GENTAMICIN_TARGETS:
        tab,_=benchmark_regressors(train,test,target,args.seed);rows.append(tab)
    pd.concat(rows,ignore_index=True).to_csv(out/'traditional_regression_benchmark.csv',index=False)
    print('Traditional benchmark saved. RNN training is exposed in notebooks/02_soft_sensor_regression.ipynb')
    print('because TensorFlow is an optional heavyweight dependency.')
if __name__=='__main__': main()
