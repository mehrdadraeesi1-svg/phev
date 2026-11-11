import pandas as pd, numpy as np, glob
from fig_common import *
def load(pat):
    return [pd.read_csv(f) for f in sorted(glob.glob(pat))]
A=load('trained_models_v3_2M/GH-SAC-MA_s4[234]_*/training_log.csv'); B=load('trained_models_v3_g999/*/training_log.csv')
grid=np.linspace(2e4,2e6,200)
def curve(dfs,col):
    ys=[np.interp(grid,d.step.values,d[col].rolling(25,min_periods=5).mean().values) for d in dfs]; ys=np.array(ys); return ys.mean(0),ys.min(0),ys.max(0)
fig,axs=plt.subplots(1,4,figsize=(8.6,2.4),gridspec_kw=dict(wspace=0.42))
for ax,(col,lab) in zip(axs,[('reward','Episode return'),('final_soc','Final SOC (%)'),('ser_pct','SERIES share (%)'),('par_pct','PARALLEL share (%)')]):
    for dfs,c,l in [(A,C['orange'],'$\\gamma$ = 0.99'),(B,C['violet'],'$\\gamma$ = 0.999')]:
        m,lo,hi=curve(dfs,col); ax.plot(grid/1e6,m,color=c,label=l,lw=1.6); ax.fill_between(grid/1e6,lo,hi,color=c,alpha=0.18,lw=0)
    ax.set_title(lab,loc='left',fontsize=8); ax.set_xlabel('Training steps (M)')
axs[0].legend(fontsize=7)
save(fig,'fig_gamma_learning'); print('ok')
