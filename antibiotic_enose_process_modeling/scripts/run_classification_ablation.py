#!/usr/bin/env python
from pathlib import Path
import argparse
import pandas as pd
from antibiotic_enose.io import load_stage_batches
from antibiotic_enose.preprocessing import make_classification_windows
from antibiotic_enose.classification.tsfcnet import TSFCAblationNet,train_torch_classifier


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--window',type=int,default=60);ap.add_argument('--epochs',type=int,default=200);ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--learning-rate',type=float,default=1e-3);ap.add_argument('--seed',type=int,default=20260913)
    args=ap.parse_args();df=load_stage_batches(args.data_dir);X,y,*_=make_classification_windows(df,args.window)
    specs=[('baseline','baseline','none'),('time_only','time','avgmax'),('sensor_only','sensor','avgmax'),('time_GAP','time','avg'),('time_GMP','time','max'),('sensor_GAP','sensor','avg'),('sensor_GMP','sensor','max'),('both','both','avgmax')]
    rows=[]
    for label,mode,pool in specs:
        model=TSFCAblationNet(window=args.window,n_sensors=16,n_classes=len(set(y.tolist())),mode=mode,pool_mode=pool)
        _,_,met,_,_,_=train_torch_classifier(model,X,y,seed=args.seed,epochs=args.epochs,batch_size=args.batch_size,learning_rate=args.learning_rate)
        rows.append({'Ablation':label,**met})
    pd.DataFrame(rows).to_csv(args.output,index=False);print('Saved',args.output)
if __name__=='__main__':main()
