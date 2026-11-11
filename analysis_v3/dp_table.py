import pandas as pd, numpy as np
K=245*17.5/100
dp=pd.concat([pd.read_csv('dp_v3_soc50.csv'),pd.read_csv('dp_v3_soc70.csv')])
e=pd.read_csv('ecms_causal_full.csv'); b=pd.read_csv('results_v3/baselines.csv'); b=b[b.controller=='RB']
base=pd.concat([b,e]); base['cs']=base.total_fuel_g-K*(base.final_soc-base.soc0)
ref=base.pivot_table(index=['cycle','soc0'],columns='controller',values='cs').reset_index()
d=pd.read_csv('/tmp/claude-0/all_runs_v3.csv')
ID={'CS-SAC':'T/WS','DH-SAC':'A/Sep','DH-SAC-EFC':'A/EFC','DH-SAC-MA':'A/MA','GH-SAC-SR':'G/Sep','GH-SAC':'G/EFC','GH-SAC-MA':'G/MA','GH-SAC-Eco':'G/EFC120'}
d['id']=d.scenario.map(ID)+np.where(d.budget=='2M',' (2M)','')
sac=d.pivot_table(index=['cycle','soc0'],columns='id',values='cs',aggfunc='mean').reset_index()
t=dp.merge(ref,on=['cycle','soc0']).merge(sac[['cycle','soc0','G/MA (2M)','A/MA (2M)','T/WS']],on=['cycle','soc0'])
t=t.rename(columns={'dp_model_cs':'DP','dw0_cs':'DP-plant','dw10_cs':'DP-plant-dwell'})
cols=['DP-plant','DP-plant-dwell','A-ECMS','RB','G/MA (2M)','A/MA (2M)','T/WS']
for c in cols: t['gap_'+c]=100*(t[c]-t.DP)/t.DP
pd.set_option('display.width',250)
out=t[['soc0','cycle','DP']+cols+['gap_'+c for c in cols]].round(1)
print(out.to_string(index=False))
agg=t.groupby('soc0')[['DP']+cols].sum()
g=agg.copy()
for c in cols: g['gap_'+c]=100*(agg[c]-agg.DP)/agg.DP
print(g.round(1).to_string())
print(t.groupby('soc0')[['gap_'+c for c in cols]].mean().round(1))
out.to_csv('table_dp.csv',index=False)
print(dp[['cycle','soc0','dw0_socf','dw10_socf','dw10_par','dw10_ser','dw10_eo','dw10_ev','dw10_sw','dw10_veto']].round(1).to_string(index=False))
