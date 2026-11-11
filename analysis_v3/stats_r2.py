import pandas as pd, numpy as np, statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
rng=np.random.default_rng(0)
d=pd.read_csv('/tmp/claude-0/all_runs_v3.csv'); d=d[d.budget=='0.5M']
M={'DH-SAC':('A','Sep'),'GH-SAC-SR':('G','Sep'),'DH-SAC-EFC':('A','EFC'),'GH-SAC':('G','EFC'),'DH-SAC-MA':('A','MA'),'GH-SAC-MA':('G','MA')}
d=d[d.scenario.isin(M)].copy(); d['act']=d.scenario.map(lambda s:M[s][0]); d['rew']=d.scenario.map(lambda s:M[s][1])
d['run']=d.scenario+'_'+d.seed.astype(str)
d['rel']=100*d.cs/d.groupby(['cycle','soc0']).cs.transform('mean')   # % of cycle mean
out={}
# 1) run-level two-way ANOVA per SOC0 (as in paper) + residual diagnostics
for s in (50.,70.):
    r=d[d.soc0==s].groupby(['run','act','rew']).cs.mean().reset_index()
    m=smf.ols('cs~C(act)*C(rew)',r).fit(); a=sm.stats.anova_lm(m,typ=2); a['eta2']=a.sum_sq/a.sum_sq.sum()
    sh=stats.shapiro(m.resid).pvalue; lev=stats.levene(*[g.cs for _,g in r.groupby(['act','rew'])]).pvalue
    print(f'SOC {s}: ANOVA\n',a.round(4),'\n Shapiro p',round(sh,3),' Levene p',round(lev,3))
    # permutation tests (10000) for main effects (restricted permutations)
    def F(df,term):
        mm=smf.ols('cs~C(act)*C(rew)',df).fit(); return sm.stats.anova_lm(mm,typ=2).loc[term,'F']
    Fr,Fa=F(r,'C(rew)'),F(r,'C(act)'); cr=ca=0; N=2000
    for _ in range(N):
        q=r.copy(); q['rew']=q.groupby('act').rew.transform(lambda x: rng.permutation(x.values)); cr+=F(q,'C(rew)')>=Fr
        q=r.copy(); q['act']=q.groupby('rew').act.transform(lambda x: rng.permutation(x.values)); ca+=F(q,'C(act)')>=Fa
    # bootstrap eta2 (resample runs within cells)
    e=[]
    for _ in range(2000):
        q=pd.concat([g.sample(len(g),replace=True,random_state=int(rng.integers(1e9))) for _,g in r.groupby(['act','rew'])])
        if q.cs.nunique()<4: continue
        mm=smf.ols('cs~C(act)*C(rew)',q).fit(); aa=sm.stats.anova_lm(mm,typ=2); e.append((aa.sum_sq/aa.sum_sq.sum())[['C(rew)','C(act)']].values)
    e=np.array(e)
    print(f' perm p reward {(cr+1)/(N+1):.4f}  action {(ca+1)/(N+1):.4f}')
    print(' bootstrap 95% CI eta2 reward',np.percentile(e[:,0],[2.5,97.5]).round(2),' action',np.percentile(e[:,1],[2.5,97.5]).round(2))
    out[s]=dict(perm_rew=(cr+1)/(N+1),perm_act=(ca+1)/(N+1),ci_rew=np.percentile(e[:,0],[2.5,97.5]),ci_act=np.percentile(e[:,1],[2.5,97.5]),shapiro=sh,levene=lev)
# 2) mixed-effects on cycle-level relative fuel: fixed act*rew*soc0, cycle fixed block, random intercept per run
m=smf.mixedlm('rel~C(act)*C(rew)+C(soc0)+C(cycle)',d,groups=d['run']).fit(reml=True)
print(m.summary().tables[1].loc[[i for i in m.summary().tables[1].index if 'act' in i or 'rew' in i]])
# Wald tests for main effects
import itertools
for term in ['C(rew)','C(act)','C(act):C(rew)']:
    names=[n for n in m.fe_params.index if n.startswith(term) and (term!='C(act)' or ':' not in n) and (term!='C(rew)' or not n.startswith('C(act)'))]
    if term=='C(rew)': names=[n for n in m.fe_params.index if n.startswith('C(rew)')]
    if term=='C(act)': names=[n for n in m.fe_params.index if n.startswith('C(act)') and ':' not in n]
    if term=='C(act):C(rew)': names=[n for n in m.fe_params.index if ':' in n]
    R=np.array([[1.0 if n==c else 0.0 for n in m.fe_params.index] for c in names])
    w=m.wald_test(R,scalar=True); print(term,'Wald chi2',round(float(w.statistic),2),'p',round(float(w.pvalue),4))
print('run random-effect var',round(float(m.cov_re.iloc[0,0]),3),' residual var',round(m.scale,3))
