"""Post-hoc paired uncertainty summary of fixed reconstruction predictions.

No model fitting or inference is performed. Resampling is within cohort,
at patient level. Intervals are pointwise percentile intervals, not simultaneous
or multiplicity-adjusted evidence of clinical utility. Cached predictions and
baseline fits are treated as fixed; training uncertainty is not included.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent/'outputs/revision_20260901')
ap.add_argument('--out',type=Path,default=None)
args=ap.parse_args(); root=args.root
out=args.out or root/'claim_support';out.mkdir(parents=True,exist_ok=True)
baseline=pd.read_parquet(root/'pixel_null_v2/baselines_celllevel.parquet')
selection=pd.read_csv(root/'pixel_null_v2/task2_window_selection_72.csv')
key='key' if 'key' in baseline else 'window'
data=baseline.drop(columns=['patient','cohort'],errors='ignore').merge(selection[['key','patient','cohort']],left_on=key,right_on='key',validate='many_to_one')
perm=pd.read_csv(root/'pixel_null_v2/permutation_results.csv').set_index('marker')
rows=[];rng=np.random.default_rng(20260905);B=10000
for marker,d in data.groupby('marker',sort=True):
    d=d.copy();d['delta']=d.r_virtues-d.r_ridge_lowo
    pts=d.groupby(['cohort','patient']).delta.mean().reset_index()
    draws=np.zeros(B);total=0
    for _,co in pts.groupby('cohort'):
        x=co.delta.to_numpy();n=len(x)
        draws+=x[rng.integers(0,n,size=(B,n))].sum(axis=1);total+=n
    draws/=total;lo,hi=np.quantile(draws,[.025,.975]);delta=pts.delta.mean()
    category='positive_interval' if lo>0 else ('negative_interval' if hi<0 else 'interval_crosses_zero')
    rows.append({'marker':marker,'n_patients':total,'n_windows':len(d),'mean_r_virtues':d.r_virtues.mean(),'mean_r_ridge_lowo':d.r_ridge_lowo.mean(),'paired_mean_delta':delta,'pointwise_ci95_low':lo,'pointwise_ci95_high':hi,'interval_category':category,'spatial_null_p':perm.loc[marker,'p_one_sided_addone'],'spatial_null_fdr_pass':bool(perm.loc[marker,'significant_FDR05'])})
result=pd.DataFrame(rows);result.to_csv(out/'marker_claim_support.csv',index=False)
metadata={'analysis_date':'2026-09-05','status':'post-hoc descriptive analysis','seed':20260905,'bootstrap_resamples':B,'unit':'patient; stratified by cohort','interval':'pointwise percentile 95%; no multiplicity correction','fits':'fixed cached model predictions and LOWO baseline predictions; no refitting','counts':result.interval_category.value_counts().to_dict(),'warning':'A positive interval is not a panel-removal recommendation or prospective validation.'}
(out/'method.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(json.dumps(metadata,indent=2))
