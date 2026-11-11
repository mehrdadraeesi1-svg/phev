import numpy as np, pandas as pd
from fig_common import *
E_act=319.18*50/1000   # kWh (OCV at SOC 0.5 x 50 Ah)
k_nom=245.0; E_nom=17.5
def s_dis(soc,kind,wd=200.0):
    sg=np.sign(0.5-soc); quad=2*4.0*(0.5-soc)/E_act
    if kind=="sep": a=0.12/3; el=0.03*3600/80
    elif kind in("efc","eco"): a=0.12/5; el=170*0.12/5
    elif kind=="aligned": return np.full_like(soc,k_nom*E_nom/E_act)
    if kind=="eco": wd=120.0
    return (wd/E_act*sg+quad+el)/a
def s_chg(soc,kind,wd=200.0):
    sg=np.sign(0.5-soc); quad=2*4.0*(0.5-soc)/E_act
    if kind=="sep": a=0.12/3
    elif kind in("efc","eco"): a=0.12/5
    elif kind=="aligned": return np.full_like(soc,k_nom*E_nom/E_act)
    if kind=="eco": wd=120.0
    return (wd/E_act*sg+quad)/a
rows=[]
for kind,lab in [("sep","Separate fuel + electricity (S2)"),("efc","Equivalent fuel consumption (S3)"),("eco","EFC, $w_\\Delta$=120 (S4)"),("aligned","Metric-anchored")]:
    for soc in (0.45,0.55):
        rows.append(dict(reward=kind,soc=soc,s_discharge=float(s_dis(np.array([soc]),kind)[0]),s_charge=float(s_chg(np.array([soc]),kind)[0])))
t=pd.DataFrame(rows); t.to_csv("figs/implied_s_table.csv",index=False); print(t.round(0))
soc=np.linspace(0.30,0.70,801)
fig,ax=plt.subplots(figsize=(5.2,3.2))
ser=[("sep",C["blue"],"Sep (w$_\\Delta$=200)"),("efc",C["orange"],"EFC (w$_\\Delta$=200)"),("eco",C["aqua"],"EFC$_{120}$ (w$_\\Delta$=120)"),("aligned",C["violet"],"Metric-anchored (MA)")]
for kind,col,lab in ser:
    y=s_dis(soc,kind); y[np.abs(soc-0.5)<1e-9]=np.nan
    ax.plot(soc*100,y,color=col,label=lab)
ax.text(69.8,k_nom*E_nom/E_act-45,"plant replenishment cost, 269 g/kWh",ha="right",color=C["violet"],fontsize=7.5)
ax.axhline(0,color=INK2,lw=0.8)
ax.set_xlabel("SOC (%)"); ax.set_ylabel("Implied equivalence factor (g/kWh)")
ax.set_xlim(30,70)
ax.legend(loc="upper right",fontsize=7.5,ncol=1,bbox_to_anchor=(1.0,1.0))

save(fig,"fig_implied_s")
