#!/usr/bin/env python
from pathlib import Path
import argparse,json
from antibiotic_enose.io import load_stage_batches
from antibiotic_enose.preprocessing import make_classification_windows
from antibiotic_enose.classification.lstm import fit_lstm_classifier

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',required=True);ap.add_argument('--output-dir',required=True)
    ap.add_argument('--window',type=int,default=60);ap.add_argument('--epochs',type=int,default=200);ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--learning-rate',type=float,default=1e-3);ap.add_argument('--seed',type=int,default=20260913)
    args=ap.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    df=load_stage_batches(args.data_dir);X,y,*_=make_classification_windows(df,args.window)
    model,hist,met,_,_=fit_lstm_classifier(X,y,seed=args.seed,learning_rate=args.learning_rate,epochs=args.epochs,batch_size=args.batch_size)
    model.save(out/'LSTM_stage_classifier.keras');(out/'LSTM_stage_metrics.json').write_text(json.dumps(met,indent=2),encoding='utf-8')
    print('Saved LSTM classifier to',out)
if __name__=='__main__':main()
