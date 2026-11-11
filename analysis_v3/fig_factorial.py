import pandas as pd, numpy as np
from fig_common import *
pr=pd.read_csv("per_run_v3.csv")
F={'DH-SAC':('argmax','sep'),'DH-SAC-EFC':('argmax','efc'),'DH-SAC-MA':('argmax','aligned'),'GH-SAC-SR':('gumbel','sep'),'GH-SAC':('gumbel','efc'),'GH-SAC-MA':('gumbel','aligned')}
X={'sep':0,'efc':1,'aligned':2}; XL=['Separate\ncosts','EFC','Metric-\nanchored']
col={'argmax':C['blue'],'gumbel':C['orange']}; off={'argmax':-0.09,'gumbel':0.09}
fig,axs=plt.subplots(1,3,figsize=(7.4,2.8),gridspec_kw=dict(wspace=0.32))
for ax,(var,soc,yl) in zip(axs,[('cs',50,'(a) CS fuel (g), SOC$_0$ = 50 %'),('cs',70,'(b) CS fuel (g), SOC$_0$ = 70 %'),('par',50,'(c) PARALLEL share (%), SOC$_0$ = 50 %')]):
    for act in ('argmax','gumbel'):
        xs=[];ms=[]
        for sc,(a,r) in F.items():
            if a!=act: continue
            f=pr[(pr.budget=='0.5M')&(pr.scenario==sc)&(pr.soc0==soc)][var]
            x=X[r]+off[act]; ax.scatter([x]*len(f),f,s=12,color=col[act],alpha=0.45,lw=0)
            m_,sd_=f.mean(),f.std(); ax.errorbar(x,m_,yerr=[[min(sd_,m_) if var=='par' else sd_],[sd_]],fmt='o',ms=5,color=col[act],capsize=2,lw=1.2)
            xs.append(x);ms.append(f.mean())
        o=np.argsort(xs); ax.plot(np.array(xs)[o],np.array(ms)[o],color=col[act],lw=1.2,label=('argmax' if act=='argmax' else 'Gumbel-max')+' (0.5 M)')
        sc2='DH-SAC-MA' if act=='argmax' else 'GH-SAC-MA'
        f2=pr[(pr.budget=='2M')&(pr.scenario==sc2)&(pr.soc0==soc)][var]
        m2,sd2=f2.mean(),f2.std(); ax.errorbar(2+off[act]*3,m2,yerr=[[min(sd2,m2) if var=='par' else sd2],[sd2]],fmt='D',ms=5,mfc='white',color=col[act],capsize=2,lw=1.2,label=('argmax' if act=='argmax' else 'Gumbel-max')+' (2 M)')
    if var=='cs':
        b=pr  # baselines
        import pandas as _p
        bl=_p.read_csv("results_v3/baselines.csv"); K=245*17.5/100; bl['cs']=bl.total_fuel_g-K*(bl.final_soc-bl.soc0)
        for ctrl,ls in (('RB',(0,(4,3))),('A-ECMS',(0,(1,2)))):
            v=bl[(bl.controller==ctrl)&(bl.soc0==soc)].cs.mean(); ax.axhline(v,color=INK2,lw=1,ls=ls)
            ax.text(-0.3,v+3,ctrl,color=INK2,fontsize=7,va='bottom')
    else:
        bl=pd.read_csv("results_v3/baselines.csv")
        v=bl[(bl.controller=='A-ECMS')&(bl.soc0==soc)].parallel_percent.mean(); ax.axhline(v,color=INK2,lw=1,ls=(0,(1,2))); ax.text(-0.3,v-0.3,'A-ECMS',color=INK2,fontsize=7,va='top')
    ax.set_xticks([0,1,2]); ax.set_xticklabels(XL); ax.set_xlim(-0.35,2.75); ax.set_title(yl,loc='left',fontsize=8,fontweight='normal')
    ax.grid(axis='x',visible=False)
h,l=axs[0].get_legend_handles_labels()
fig.legend(h,l,loc='upper center',ncol=4,fontsize=7.5,bbox_to_anchor=(0.5,1.06))
save(fig,"fig_factorial")
