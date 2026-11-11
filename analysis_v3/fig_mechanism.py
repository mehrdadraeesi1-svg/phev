import pandas as pd, numpy as np
from fig_common import *
sb=pd.read_csv('mode_by_speed.csv')
MC={'EV':C['blue'],'SERIES':C['orange'],'PARALLEL':C['aqua'],'ENGINE_ONLY':C['yellow']}
ctrls=[('RB','RB'),('A-ECMS','A-ECMS'),('DH-SAC-MA (2M)','A/MA'),('GH-SAC-MA (2M)','G/MA')]
bands=['<50','50-80','>80']; BL={'<50':'< 50 km/h','50-80':'50–80 km/h','>80':'> 80 km/h'}
fig,axs=plt.subplots(1,3,figsize=(7.4,2.7),sharey=True,gridspec_kw=dict(wspace=0.12))
for ax,b in zip(axs,bands):
    for i,(c,lab) in enumerate(ctrls):
        r=sb[(sb.ctrl==c)&(sb.band==b)].iloc[0]; bottom=0
        for m in ['EV','SERIES','PARALLEL','ENGINE_ONLY']:
            v=r[m]; ax.bar(i,v,bottom=bottom,color=MC[m],width=0.7,edgecolor='white',linewidth=1,label=m if (i==0 and b=='<50') else None)
            if v>=8: ax.text(i,bottom+v/2,f"{v:.0f}",ha='center',va='center',fontsize=6.8,color='white' if m in('EV','PARALLEL') else INK)
            bottom+=v
    ax.set_xticks(range(4)); ax.set_xticklabels([l for _,l in ctrls],fontsize=7.5); ax.set_title(BL[b],loc='left',fontsize=8,fontweight='normal')
    ax.grid(axis='x',visible=False); ax.set_ylim(0,100)
axs[0].set_ylabel("Share of traction time (%)")
fig.legend(loc='upper center',ncol=4,fontsize=7.5,bbox_to_anchor=(0.5,1.07),labels=['EV','SERIES','PARALLEL','ENGINE_ONLY'],handles=[plt.Rectangle((0,0),1,1,color=MC[m]) for m in MC])
save(fig,"fig_mode_by_speed")

L=pd.read_pickle('logs_v3/logs_soc50.pkl')
h=L[(L.cycle=='highway')]
sel=[('A-ECMS',0,'A-ECMS'),('GH-SAC-MA (2M)',42,'G/MA (seed 42)'),('RB',0,'RB')]
t0,t1=0,800
fig,axs=plt.subplots(5,1,figsize=(7.4,5.2),sharex=True,gridspec_kw=dict(height_ratios=[1.1,0.35,0.35,0.35,1.1],hspace=0.25))
g=h[(h.ctrl=='A-ECMS')].sort_values('time'); g=g[(g.time>=t0)&(g.time<=t1)]
axs[0].plot(g.time,g.velocity_kmh,color=INK,lw=1); axs[0].set_ylabel("Speed\n(km/h)")
code={'EV':0,'SERIES':1,'PARALLEL':2,'ENGINE_ONLY':3,'REGEN':4}
MC2=dict(MC); MC2['REGEN']=GRID
for ax,(c,s,lab) in zip(axs[1:4],sel):
    g=h[(h.ctrl==c)&(h.seed==s)].sort_values('time'); g=g[(g.time>=t0)&(g.time<=t1)]
    for m,colr in MC2.items():
        x=g.time[g['mode']==m]; ax.vlines(x,0,1,color=colr,lw=1.2)
    ax.set_yticks([]); ax.set_ylabel(lab,rotation=0,ha='right',va='center',fontsize=7.5); ax.grid(False)
    for sp in ('left',): ax.spines[sp].set_visible(False)
for (c,s,lab),colr in zip(sel,[C['violet'],C['orange'],INK2]):
    g=h[(h.ctrl==c)&(h.seed==s)].sort_values('time'); g=g[(g.time>=t0)&(g.time<=t1)]
    axs[4].plot(g.time,g.soc,color=colr,lw=1.5,label=lab)
axs[4].set_ylabel("SOC (%)"); axs[4].set_xlabel("Time (s)"); axs[4].legend(fontsize=7,ncol=3,loc='upper left')
fig.legend(handles=[plt.Rectangle((0,0),1,1,color=MC2[m]) for m in MC2],labels=list(MC2),loc='upper center',ncol=5,fontsize=7.3,bbox_to_anchor=(0.5,0.95))
save(fig,"fig_highway_trace")
