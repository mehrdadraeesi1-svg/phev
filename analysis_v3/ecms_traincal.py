import numpy as np, pandas as pd, PHEVEnv_v13 as E, ecms_v13 as A
K=245*17.5/100
b=pd.read_csv('results_v3/baselines.csv'); t=b[(b.controller=='A-ECMS')&b.cycle.isin(E.TRAIN_CYCLES)]
s0=float(np.median(t.s0)); print('training-only s0 (median of 8 tuned values):', s0, 'mean', t.s0.mean())
rows=[]
for n,cd in E.load_cycles(list(E.CYCLE_FILES)).items():
    env=E.PHEVMultiModeEnv([cd]); c=A.precompute(env,cd)
    for soc0 in (0.5,0.7):
        s=A.simulate(env,cd,c,s0,soc0); rows.append(dict(cycle=n,soc0=soc0*100,s0=s0,fuel=s['total_fuel_g'],socf=s['final_soc'],cs=s['total_fuel_g']-K*(s['final_soc']-soc0*100)))
d=pd.DataFrame(rows); d.to_csv('ecms_traincal.csv',index=False)
e=pd.read_csv('ecms_causal_full.csv'); e['cs_old']=e.total_fuel_g-K*(e.final_soc-e.soc0)
m=d.merge(e[['cycle','soc0','cs_old']]); m['d%']=100*(m.cs/m.cs_old-1); print(m.round(2).to_string())
m['ho']=~m.cycle.isin(E.TRAIN_CYCLES)
print(m.groupby('soc0')[['cs','cs_old']].mean().round(1)); print(m.groupby(['soc0','ho'])[['cs','cs_old']].sum().round(1))
