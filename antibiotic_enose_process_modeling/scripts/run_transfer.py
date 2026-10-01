#!/usr/bin/env python
from pathlib import Path
import argparse
import pandas as pd
from antibiotic_enose.constants import TRANSFER_TARGETS
from antibiotic_enose.transfer.representation import GentamicinSourceRepresentation
from antibiotic_enose.transfer.evaluation import learning_curve_for_variable,summarize_learning_curves,threshold_table


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',required=True);ap.add_argument('--output-dir',required=True)
    ap.add_argument('--n-repeats',type=int,default=100);ap.add_argument('--seed',type=int,default=20260913)
    args=ap.parse_args();D=Path(args.data_dir);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    src=pd.read_csv(D/'Gentamicin_REAL_source.csv')
    source_repr=GentamicinSourceRepresentation(6).fit(src)
    all_rows=[]
    for product,vars_ in TRANSFER_TARGETS.items():
        df=pd.read_csv(D/f'{product}_REAL_aligned.csv')
        for var in vars_:
            if var in df.columns:
                all_rows.append(learning_curve_for_variable(product,df,var,source_repr,n_repeats=args.n_repeats,seed=args.seed))
    rep=pd.concat(all_rows,ignore_index=True);rep.to_csv(out/'repeat_level.csv',index=False)
    summary=summarize_learning_curves(rep);summary.to_csv(out/'learning_curve_summary.csv',index=False)
    threshold_table(summary).to_csv(out/'r2_threshold_summary.csv',index=False)
    print('Saved outputs to',out)
if __name__=='__main__': main()
