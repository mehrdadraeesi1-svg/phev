import pandas as pd, numpy as np, glob
from fig_common import *
def load(pat):
    out={}
    for f in glob.glob(pat):
        run=f.split('/')[1]; sc=run.split('_s')[0]; out.setdefault(sc,[]).append(pd.read_csv(f))
    return out
L05=load('trained_models_v3/*/training_log.csv'); L2=load('trained_models_v3_2M/*/training_log.csv')
def curve(dfs,col,grid):
    ys=[]
    for d in dfs:
        r=d[col].rolling(25,min_periods=5).mean()
        ys.append(np.interp(grid,d.step.values,r.values,left=np.nan,right=np.nan))
    ys=np.array(ys); return np.nanmean(ys,0),np.nanmin(ys,0),np.nanmax(ys,0)
panels=[("(a) Separate costs",[("DH-SAC",C['blue'],"argmax"),("GH-SAC-SR",C['orange'],"Gumbel-max")],L05,5e5),
        ("(b) EFC",[("DH-SAC-EFC",C['blue'],"argmax"),("GH-SAC",C['orange'],"Gumbel-max"),("GH-SAC-Eco",C['aqua'],"Gumbel, $w_\\Delta$=120")],L05,5e5),
        ("(c) Metric-anchored (2 M)",[("DH-SAC-MA",C['blue'],"argmax"),("GH-SAC-MA",C['orange'],"Gumbel-max")],L2,2e6),
        ("(d) T/WS (threshold)",[("CS-SAC",C['violet'],"threshold")],L05,5e5)]
fig,axs=plt.subplots(2,4,figsize=(8.0,4.4),gridspec_kw=dict(wspace=0.35,hspace=0.35),sharex='col')
for j,(title,items,L,T) in enumerate(panels):
    g=np.linspace(0,T,300)
    for sc,col,lab in items:
        for i,var in enumerate(('reward','par_pct')):
            m,lo,hi=curve(L[sc],var,g); ax=axs[i,j]
            ax.fill_between(g/1e6,lo,hi,color=col,alpha=0.15,lw=0); ax.plot(g/1e6,m,color=col,lw=1.4,label=lab)
    axs[0,j].set_title(title,loc='left',fontsize=8)
    axs[1,j].set_xlabel("Training steps (M)")
    axs[1,j].legend(fontsize=6.3,loc="upper right")
    if j==3: pass
    elif T==5e5:
        for i in range(2): axs[i,j].axvline(0.15,color=INK2,lw=0.8,ls=(0,(2,2)))
    else:
        for i in range(2): axs[i,j].axvline(0.6,color=INK2,lw=0.8,ls=(0,(2,2)))
axs[0,0].set_ylabel("Episode return\n(rolling mean, 25 ep.)"); axs[1,0].set_ylabel("PARALLEL share during\ntraining (%)")
save(fig,"fig_learning")
