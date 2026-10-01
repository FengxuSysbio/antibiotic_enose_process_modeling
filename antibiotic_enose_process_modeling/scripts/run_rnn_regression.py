#!/usr/bin/env python
from pathlib import Path
import argparse, json
import pandas as pd
from antibiotic_enose.io import load_paired_regression_batches
from antibiotic_enose.constants import GENTAMICIN_TARGETS
from antibiotic_enose.regression.rnn import fit_rnn_regressor

SELECTED={'PMV':'BiLSTM','RS':'BiGRU','TS':'BiGRU','NH4':'BiLSTM','Titer':'LSTM'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',required=True);ap.add_argument('--external-batch',required=True)
    ap.add_argument('--output-dir',required=True);ap.add_argument('--window',type=int,default=6)
    ap.add_argument('--epochs',type=int,default=300);ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--learning-rate',type=float,default=0.001);ap.add_argument('--verbose',type=int,default=0)
    args=ap.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    df=load_paired_regression_batches(args.data_dir);train=df[df.Batch!=args.external_batch].copy();test=df[df.Batch==args.external_batch].copy()
    if test.empty:raise ValueError(f'External batch {args.external_batch!r} not found')
    rows=[]
    for target in GENTAMICIN_TARGETS:
        kind=SELECTED[target]
        model,hist,y,pred,met,_=fit_rnn_regressor(train,test,target,kind,window=args.window,
            learning_rate=args.learning_rate,epochs=args.epochs,batch_size=args.batch_size,verbose=args.verbose)
        rows.append({'Target':target,'Model':kind,**met})
        pd.DataFrame({'Observed':y,'Predicted':pred}).to_csv(out/f'{target}_{kind}_external_predictions.csv',index=False)
        pd.DataFrame(hist).to_csv(out/f'{target}_{kind}_training_history.csv',index=False)
        model.save(out/f'{target}_{kind}.keras')
    pd.DataFrame(rows).to_csv(out/'selected_rnn_external_metrics.csv',index=False)
    print('Saved RNN models and external-batch predictions to',out)
if __name__=='__main__':main()
