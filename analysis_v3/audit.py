import pandas as pd, numpy as np
L=pd.read_pickle('logs_v3/logs_soc50.pkl')
K=245*17.5/100
def aud(g):
    g=g.sort_values('time'); on=g.P_ice_kW>0
    soc0=50.0; cs=g.fuel_g.sum()-K*(g.soc.iloc[-1]-soc0)
    return pd.Series(dict(cs=cs,Eice=g.P_ice_kW.sum()/3600,bsfc=g.fuel_g.sum()/max(g.P_ice_kW.sum()/3600,1e-9),
      Edis=g.P_batt_kW.clip(lower=0).sum()/3600,Echg=-g.P_batt_kW.clip(upper=0).sum()/3600,
      Iloss=None, rms=np.sqrt((g.current_A**2).mean()),par=100*(g['mode']=='PARALLEL').mean(),ser=100*(g['mode']=='SERIES').mean(),
      eo=100*(g['mode']=='ENGINE_ONLY').mean(),ev=100*(g['mode']=='EV').mean(),starts=int((on&~on.shift(1,fill_value=False)).sum()),socf=g.soc.iloc[-1],
      ser_pdem=g.power_demand_kW[g['mode']=='SERIES'].mean(), cold=g.fuel_g[on&(g.T_engine<80)].sum()))
A=L.groupby(['ctrl','seed','cycle']).apply(aud).reset_index()
A.to_csv('energy_audit.csv',index=False)
m=A.groupby(['ctrl','cycle']).mean(numeric_only=True).reset_index()
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(A.groupby('ctrl').mean(numeric_only=True).drop(columns=['seed']).round(2))
print(m.pivot_table(index='cycle',columns='ctrl',values=['Eice','Edis','par']).round(1))
# speed bands
L['band']=pd.cut(L.velocity_kmh,[-1,1,50,80,200],labels=['stop','<50','50-80','>80'])
sb=L[L.power_demand_kW>=0].groupby(['ctrl','band'])['mode'].value_counts(normalize=True).unstack().fillna(0)*100
print(sb.round(1))
sb.to_csv('mode_by_speed.csv')
