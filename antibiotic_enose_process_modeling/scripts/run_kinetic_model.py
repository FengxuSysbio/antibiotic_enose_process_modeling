#!/usr/bin/env python
from pathlib import Path
import argparse, json
import pandas as pd
from antibiotic_enose.regression.kinetics import fit_model7, kinetic_augmentation

TARGETS=['PMV','RS','TS','NH4','Titer']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output-dir',required=True);ap.add_argument('--n-points',type=int,default=1000)
    args=ap.parse_args();p=Path(args.input);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(p) if p.suffix.lower()=='.csv' else pd.read_excel(p)
    missing=[c for c in ['Time_h',*TARGETS] if c not in df.columns]
    if missing:raise ValueError(f'Missing columns: {missing}')
    t=df.Time_h.to_numpy(float);Y=df[TARGETS].to_numpy(float)
    params,cost,pred=fit_model7(t,Y)
    pd.DataFrame(pred,columns=TARGETS).assign(Time_h=t).to_csv(out/'model7_fit_at_measured_times.csv',index=False)
    kinetic_augmentation(t,Y,params,args.n_points,TARGETS).to_csv(out/'model7_kinetic_augmentation.csv',index=False)
    (out/'model7_parameters.json').write_text(json.dumps({'parameters':params.tolist(),'least_squares_cost':float(cost)},indent=2),encoding='utf-8')
    print('Saved kinetic reconstruction to',out)
if __name__=='__main__':main()
