import pandas as pd
from fig_common import *
d=pd.read_csv("mode_diag_v3.csv")
col={'argmax':C['blue'],'gumbel':C['orange']}
ls={'sep':'-','efc':(0,(5,2)),'aligned':(0,(1,1.5))}; mk={'sep':'o','efc':'s','aligned':'^'}
RL={'sep':'Sep','efc':'EFC','aligned':'MA'}
fig,axs=plt.subplots(1,3,figsize=(7.4,2.6),gridspec_kw=dict(wspace=0.35))
for (a,r),g in d.groupby(['action','reward']):
    g=g.sort_values('step'); lab=('A' if a=='argmax' else 'G')+'/'+RL[r]
    for ax,v,sc in zip(axs,['H_adm','greedy_par_adm','frac_dQpos_adm'],[1,100,100]):
        ax.plot(g.step/1e3,g[v]*sc,color=col[a],ls=ls[r],marker=mk[r],ms=4,lw=1.5,label=lab)
T=["(a) Mode entropy under training\nbehaviour (nats)","(b) PARALLEL chosen by the\ndeterministic policy (%)","(c) States where the critic values\nforced PARALLEL above its choice (%)"]
for ax,t in zip(axs,T):
    ax.set_title(t,loc='left',fontsize=7.8,fontweight='normal'); ax.set_xlabel("Checkpoint (k steps)"); ax.set_xticks([150,300,500])
axs[2].axhline(50,color=INK2,lw=0.8,ls=(0,(2,2)))
h,l=axs[0].get_legend_handles_labels()
ordr=['A/Sep','A/EFC','A/MA','G/Sep','G/EFC','G/MA']; idx=[l.index(o) for o in ordr]; h=[h[i] for i in idx]; l=ordr
fig.legend(h,l,loc='upper center',ncol=6,fontsize=7.3,bbox_to_anchor=(0.5,1.1))
save(fig,"fig_mode_diag")
print(d.groupby(['action','reward'])[['H_adm','greedy_par_adm','frac_dQpos_adm','ecms_par_adm','dQpos_where_ecms_par']].last().round(3))
print(d[d.step==500000][['scenario','H_adm','p_par_adm','greedy_par_adm','frac_dQpos_adm','ecms_par_adm','dQpos_where_ecms_par','n_adm']].round(3))
