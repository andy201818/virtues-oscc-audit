"""Validate and summarize released result tables, not raw-image/GPU inference."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu,wilcoxon

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent/'outputs/revision_20260901');ap.add_argument('--report',type=Path,default=Path('reproduce_report.json'));a=ap.parse_args();R=a.root
checks=[]
def check(name,ok,detail):checks.append({'check':name,'pass':bool(ok),'detail':str(detail)})
def csv(path):return pd.read_csv(R/path)
bd=pd.read_parquet(R/'pixel_null_v2/baselines_celllevel.parquet')
s=bd.groupby('marker').agg(n=('r_virtues','size'),r_virtues=('r_virtues','mean'),r_ridge=('r_ridge_lowo','mean'),r_template=('r_template_lowo','mean'))
published=csv('pixel_null_v2/baselines_r.csv').set_index('marker')
for marker,row in s.iterrows():
    cols=['r_virtues','r_ridge','r_template'];err=np.max(np.abs(row[cols].to_numpy(float)-published.loc[marker,cols].to_numpy(float)))
    check('Table3 '+marker,err<1e-10 and row.n==published.loc[marker,'n'],err)
check('Marker counts',len(s)==39 and (s.n==72).sum()==37 and (s.n==36).sum()==2,s.n.value_counts().to_dict())
perm=csv('pixel_null_v2/permutation_results.csv');check('Spatial null',(perm.p_one_sided_addone==.0001).all() and perm.significant_FDR05.all(),len(perm))
f=csv('figs/fig2_v2_source_data.csv');check('Table2B macro',round(f.f1_celllevel.mean(),3)==.756,f.f1_celllevel.mean())
t=json.loads((R/'task1/kao1_cis.json').read_text());b=json.loads((R/'task1/baseline_results.json').read_text());c=json.loads((R/'task1/concat_results.json').read_text())
check('Token patient mean',round(t['patient_groupkfold5_macro_f1_mean'],3)==.713,t['patient_groupkfold5_macro_f1_mean'])
check('Fusion patient mean',round(c['patient_groupkfold5'][0],3)==.801,c['patient_groupkfold5'][0])
bf=[b['patient_groupkfold5_macro_f1_mean'],b['patient_groupkfold5_macro_f1_sd'],b['patient_groupkfold5_accuracy_mean'],b['patient_groupkfold5_folds_f1']];cf=c['patient_groupkfold5']
check('Intensity patient mean',round(bf[0],3)==.783,bf[0])
check('Fusion paired patient folds',np.all(np.array(cf[3])>np.array(bf[3])),np.array(cf[3])-np.array(bf[3]))
w=csv('pixel_null_v2/task2_window_selection_72.csv');check('Task2 units',(len(w),w.tissue.nunique(),w.patient.nunique())==(72,36,36),[len(w),w.tissue.nunique(),w.patient.nunique()])
d=csv('pixel_null_v2/task2_roi_representativeness.csv');sel=d[d.selected];exc=d[~d.selected]
for field,ref in [('n_cells',.73),('vessel_frac',.78)]:
    p=mannwhitneyu(sel[field],exc[field],alternative='two-sided').pvalue;check('Selection '+field,round(p,2)==ref,p)
st=csv('moduleB/cluster_stability.csv');med=st.loc[st.seed!=0,'ari_vs_primary'].median();check('Cluster stability',round(med,3)==.407,med)
ki=csv('moduleB/ki67_patient_paired.csv')
for co,ref in [('UOP',-.001),('STA',-.012)]:
    x=ki.loc[ki.cohort==co,'diff'];check('Ki67 '+co,round(x.median(),3)==ref,{'median':x.median(),'P':wilcoxon(x).pvalue})
th=csv('moduleB/threshold_sensitivity.csv');check('Negative thresholds',(th.median_diff_frac_ki67pos<0).sum()==10,len(th))
oc=csv('moduleB/outcome_analysis.csv');check('Outcome sample',(oc.n==23).all() and (oc.n_events==9).all(),oc[['n','n_events']].values.tolist())
for i,ref in enumerate([.5415365837364132,.6540852217724199]):check('Outcome OR '+str(i),abs(oc.iloc[i].OR_per_1SD-ref)<1e-10,oc.iloc[i].OR_per_1SD)
v=csv('moduleB/diag_vessel_axis_seed_consistency.csv');check('Vessel-axis range',(round(v.max_vessel_pct.min(),2),round(v.max_vessel_pct.max(),2))==(.88,.96),[v.max_vessel_pct.min(),v.max_vessel_pct.max()])
lp=csv('moduleB/lopo_vs_lowo_by_marker.csv');check('LOPO positive-mean set',set(lp.loc[lp.r_virtues>lp.r_ridge_lopo,'marker'])==set(lp.loc[lp.r_virtues>lp.r_ridge_lowo,'marker']),int((lp.r_virtues>lp.r_ridge_lopo).sum()))
result={'scope':'table-level aggregation and selected consistency checks; not raw-data or GPU reproduction','n_pass':sum(x['pass'] for x in checks),'n_total':len(checks),'checks':checks}
a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2),encoding='utf-8');print(result['n_pass'],result['n_total'])
for x in checks:
    if not x['pass']:print(x)
raise SystemExit(0 if result['n_pass']==result['n_total'] else 1)
