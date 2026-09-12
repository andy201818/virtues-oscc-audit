"""Recompute input moments from ROI sufficient statistics; no inference or refitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
Z=ROOT/'outputs/revision_20260901/zscore_sensitivity'
d=pd.read_parquet(Z/'roi_channel_stats.parquet')
d['patient']=d.roi.str.extract(r'^([A-Za-z]+\d+)',expand=False)
def moments(x):
    a=x.groupby('marker')[['sum','sq','n']].sum()
    mean=a['sum']/a['n']
    return pd.DataFrame({'mean':mean,'std':np.sqrt(a['sq']/a['n']-mean**2)})
pooled=moments(d)
folds=json.loads((Z/'task1_patient_folds.json').read_text(encoding='utf-8'))
rows=[]
for f in folds:
    assert not (set(f['train_patients'])&set(f['test_patients']))
    a=moments(d.loc[d.patient.isin(f['train_patients'])]).join(pooled,rsuffix='_pooled')
    rows.append({'fold':f['fold'],'max_abs_mean_shift_in_SD':float(((a['mean']-a.mean_pooled)/a.std_pooled).abs().max()),'max_abs_std_ratio_dev':float((a['std']/a.std_pooled-1).abs().max())})
result=pd.DataFrame(rows)
ref=pd.read_csv(Z/'zscore_task1_actual_folds_summary.csv').sort_values('scenario')
for c in ['max_abs_mean_shift_in_SD','max_abs_std_ratio_dev']:
    assert np.allclose(result[c],ref[c],rtol=1e-10,atol=1e-12),c
print(result.to_string(index=False))
print('Input-moment summaries reproduced. Model-output sensitivity remains unmeasured.')
