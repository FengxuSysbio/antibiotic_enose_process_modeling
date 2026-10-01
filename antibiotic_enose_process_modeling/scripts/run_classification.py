#!/usr/bin/env python
from pathlib import Path
import argparse, json
import pandas as pd
import numpy as np
from antibiotic_enose.io import load_stage_batches
from antibiotic_enose.preprocessing import make_classification_windows
from antibiotic_enose.classification.features import pca_tsne_embedding
from antibiotic_enose.classification.baselines import benchmark_classifiers
from antibiotic_enose.classification.tsfcnet import TSFCNet, train_torch_classifier
from antibiotic_enose.classification.anomaly import fit_mahalanobis_reference, mahalanobis_scores


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',required=True);ap.add_argument('--output-dir',required=True)
    ap.add_argument('--window',type=int,default=60);ap.add_argument('--stride',type=int,default=1)
    ap.add_argument('--epochs',type=int,default=200);ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--learning-rate',type=float,default=1e-3);ap.add_argument('--optimizer',choices=['adam','adamw'],default='adam')
    ap.add_argument('--seed',type=int,default=20260913)
    args=ap.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    df=load_stage_batches(args.data_dir)
    X,y,batch,t,abn=make_classification_windows(df,args.window,args.stride)
    Zp,Zt,var=pca_tsne_embedding(X, seed=args.seed)
    pd.DataFrame(Zp,columns=['PC1','PC2']).assign(Stage=y,Batch=batch,Time_h=t).to_csv(out/'pca_embedding.csv',index=False)
    pd.DataFrame(Zt,columns=['tSNE1','tSNE2']).assign(Stage=y,Batch=batch,Time_h=t).to_csv(out/'tsne_embedding.csv',index=False)
    baseline,_,_=benchmark_classifiers(X,y,seed=args.seed);baseline.to_csv(out/'baseline_classification_metrics.csv',index=False)
    model=TSFCNet(window=args.window,n_sensors=16,n_classes=len(np.unique(y)))
    model,hist,metrics,_,_,split=train_torch_classifier(model,X,y,seed=args.seed,epochs=args.epochs,
        batch_size=args.batch_size,learning_rate=args.learning_rate,optimizer_name=args.optimizer)
    pd.DataFrame({'epoch':np.arange(1,len(hist)+1),'loss':hist}).to_csv(out/'tsfc_training_loss.csv',index=False)
    (out/'tsfc_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    # anomaly reference: use only windows whose endpoint is not abnormal
    normal=abn==0
    if np.any(~normal):
        ref,thr,_=fit_mahalanobis_reference(X[normal])
        scores=mahalanobis_scores(ref,X)
        pd.DataFrame({'Batch':batch,'Time_h':t,'Is_abnormal':abn,'Mahalanobis':scores,
                      'Threshold':thr}).to_csv(out/'mahalanobis_scores.csv',index=False)
    print('Saved outputs to',out)
if __name__=='__main__': main()
