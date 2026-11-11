import pandas as pd, numpy as np
import statsmodels.formula.api as smf, statsmodels.api as sm
pd.set_option('display.width',260); pd.set_option('display.max_columns',40)
K=245*17.5/100
d=pd.read_csv('/tmp/claude-0/all_runs_v3.csv')
b=pd.read_csv('results_v3/baselines.csv'); b=b[b.controller=='RB']
e=pd.read_csv('ecms_causal_full.csv')
base=pd.concat([b,e]); base['cs']=base.total_fuel_g-K*(base.final_soc-base.soc0)
ref=base.pivot_table(index=['cycle','soc0'],columns='controller',values='cs').reset_index().rename(columns={'A-ECMS':'ec','RB':'rb'})
d=d.drop(columns=['ec','rb']).merge(ref,on=['cycle','soc0'])
ID={'CS-SAC':'T/WS','DH-SAC':'A/Sep','DH-SAC-EFC':'A/EFC','DH-SAC-MA':'A/MA','GH-SAC-SR':'G/Sep','GH-SAC':'G/EFC','GH-SAC-MA':'G/MA','GH-SAC-Eco':'G/EFC120'}
d['id']=d.scenario.map(ID)+np.where(d.budget=='2M',' (2M)','')
agg=dict(cs=('cs','mean'),rb=('rb','mean'),ec=('ec','mean'),socf=('final_soc','mean'),ev=('ev_percent','mean'),ser=('series_percent','mean'),par=('parallel_percent','mean'),eo=('engine_only_percent','mean'),sw=('mode_switches','mean'),starts=('engine_starts','mean'),rms=('rms_current_A','mean'),veto=('n_vetoed','mean'),ovr=('n_overridden','mean'),peakT=('peak_T_batt','mean'),q=('qloss_total','mean'))
pr=d.groupby(['id','seed','soc0']).agg(**agg).reset_index()
pr['vs_rb']=100*(pr.rb-pr.cs)/pr.rb; pr['vs_ec']=100*(pr.ec-pr.cs)/pr.ec
pr.to_csv('per_run_final.csv',index=False)
T=pr.groupby(['soc0','id']).agg(cs=('cs','mean'),sd=('cs','std'),vs_rb=('vs_rb','mean'),vs_ec=('vs_ec','mean'),ev=('ev','mean'),ser=('ser','mean'),par=('par','mean'),eo=('eo','mean'),sw=('sw','mean'),starts=('starts','mean'),socf=('socf','mean'),rms=('rms','mean'),veto=('veto','mean'),ovr=('ovr','mean'),peakT=('peakT','mean'),q=('q','mean')).round(1)
bb=base.groupby(['soc0','controller']).agg(cs=('cs','mean'),ev=('ev_percent','mean'),ser=('series_percent','mean'),par=('parallel_percent','mean'),eo=('engine_only_percent','mean'),sw=('mode_switches','mean'),starts=('engine_starts','mean'),socf=('final_soc','mean'),rms=('rms_current_A','mean'),veto=('n_vetoed','mean'),ovr=('n_overridden','mean'),peakT=('peak_T_batt','mean'),q=('qloss_total','mean')).round(1)
print(bb); print(T)
T.to_csv('table_main.csv'); bb.to_csv('table_baselines.csv')
# per cycle best
pc=d[(d.id=='G/MA (2M)')].groupby(['soc0','cycle']).cs.agg(['mean','std']).reset_index().merge(ref,on=['cycle','soc0'])
pc['vs_rb']=100*(pc.rb-pc['mean'])/pc.rb; pc['vs_ec']=100*(pc.ec-pc['mean'])/pc.ec
print(pc.round(1)); pc.to_csv('table_percycle.csv',index=False)
# held-out
d['set']=np.where(d.cycle.isin(['nedc','ftp75','us06','highway']),'train','held-out')
ho=d[d.soc0==50].groupby(['id','seed','set']).agg(cs=('cs','mean'),rb=('rb','mean'),ec=('ec','mean')).reset_index()
ho['vs_rb']=100*(ho.rb-ho.cs)/ho.rb; ho['vs_ec']=100*(ho.ec-ho.cs)/ho.ec
H=ho.groupby(['id','set'])[['vs_rb','vs_ec']].mean().round(1).unstack(); print(H); H.to_csv('table_heldout.csv')
# ANOVA
F={'A/Sep':('argmax','sep'),'A/EFC':('argmax','efc'),'A/MA':('argmax','aligned'),'G/Sep':('gumbel','sep'),'G/EFC':('gumbel','efc'),'G/MA':('gumbel','aligned')}
out=[]
for soc in (50,70):
    f=pr[pr.id.isin(F)&(pr.soc0==soc)].copy(); f['A']=f.id.map(lambda s:F[s][0]); f['B']=f.id.map(lambda s:F[s][1])
    for var in ('cs','par'):
        m=smf.ols(f'{var} ~ C(A)*C(B)',data=f).fit(); t=sm.stats.anova_lm(m,typ=2); t['eta2']=t.sum_sq/t.sum_sq.sum()
        t['soc0']=soc; t['var']=var; out.append(t.reset_index())
A_=pd.concat(out); print(A_.round(4)); A_.to_csv('table_anova.csv',index=False)
# 2M aligned: gumbel vs argmax Welch
from scipy import stats
for soc in (50,70):
    g=pr[(pr.id=='G/MA (2M)')&(pr.soc0==soc)].cs; a=pr[(pr.id=='A/MA (2M)')&(pr.soc0==soc)].cs
    print(soc, stats.ttest_ind(g,a,equal_var=False))
