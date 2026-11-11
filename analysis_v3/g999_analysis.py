import pandas as pd, glob, numpy as np
from scipy import stats
K=245*17.5/100
def load(root):
    d=pd.concat([pd.read_csv(f) for f in glob.glob(root+'/*/summary.csv')]); d['cs']=d.total_fuel_g-K*(d.final_soc-d.soc0); d['seed']=d.run.str.extract(r'_s(\d\d)_').astype(int); return d
e=pd.read_csv('ecms_causal_full.csv'); e['cs']=e.total_fuel_g-K*(e.final_soc-e.soc0)
dp=pd.concat([pd.read_csv('dp_v3_soc50.csv'),pd.read_csv('dp_v3_soc70.csv')])
ref=pd.read_csv('runs_2M_5seed_ool.csv'); ref=ref[ref.scenario=='GH-SAC-MA']
fin=pd.read_csv('runs_final_ckpt.csv'); fin=fin[(fin.scenario=='GH-SAC-MA')&(fin.budget=='2M')]
out={}
for nm,d in [('g99 best',ref),('g99 final',fin),('g999 best',load('results_v3_g999')),('g999 final',load('results_v3_g999_final'))]:
    for s in (50.,70.):
        x=d[d.soc0==s]; per=x.groupby('seed').cs.mean(); ec=e[e.soc0==s].cs.mean(); J=dp[dp.soc0==s].dp_model_cs.mean()
        print(f"{nm:11s} SOC{int(s)} cs {per.mean():7.1f}±{per.std():5.1f} vsEC {100*(per.mean()/ec-1):+5.2f}% vsDP {100*(per.mean()/J-1):+5.2f}%  socT {x.final_soc.mean():5.1f} SER {x.series_percent.mean():5.1f} PAR {x.parallel_percent.mean():4.1f} EO {x.engine_only_percent.mean():4.1f} EV {x.ev_percent.mean():5.1f} rms {x.rms_current_A.mean():4.1f} sw {x.mode_switches.mean():5.0f} term {x.termination_reason.notna().sum()}")
        out[(nm,s)]=per
for s in (50.,70.):
    for a,b in [('g99 best','g999 best'),('g99 final','g999 final')]:
        print(s,a,'vs',b,'p',round(stats.ttest_ind(out[(a,s)],out[(b,s)],equal_var=False).pvalue,3))
