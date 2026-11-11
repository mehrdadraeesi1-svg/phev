"""Headroom test for learning the equivalence factor: A-ECMS whose s(t) is taken from the
DP costate (non-causal oracle), s = -dJ_{k+1}/dSOC / E_chem, under the SAME candidate set,
supervisor and 10-s dwell as causal A-ECMS. Gives an upper bound for any controller that
only adapts s(t) on top of A-ECMS."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'.')
import PHEVEnv_v13 as E, ecms_v13 as A, dp_v3 as D
K = D.K_G_PER_PCT
def run(n, soc0=0.5, clip=(100., 450.)):
    cd=E.load_cycles([n])[n]
    env=E.PHEVMultiModeEnv([cd], min_dwell=0); env.reset(options={"soc_init":soc0})
    grid,J,j0,_=D.solve(env,cd,soc0)
    b=env._batt; Echem=b.C_pack*b.get_ocv(0.5,30.0)/1000.0   # kWh per unit SOC
    envd=E.PHEVMultiModeEnv([cd]); cands=A.precompute(envd,cd)
    def sim(mode):
        envd.reset(options={"soc_init":soc0}); done=False; ss=[]
        while not done:
            k=envd.steps; cs=cands[k] if k<len(cands) else None
            if cs is None: a={"mode":0,"split":np.array([0.5],np.float32)}
            else:
                if envd.dwell<envd.min_dwell and envd.current_mode!=E.DriveMode.REGEN:
                    cs=[c for c in cs if c[0]==envd.current_mode.value] or cs
                if mode=='dp':
                    soc=envd.soc; d=2e-3
                    g=(np.interp(soc+d,grid,J[k+1])-np.interp(soc-d,grid,J[k+1]))/(2*d)
                    s=float(np.clip(-g/Echem,*clip))
                else:
                    s=258.0*(1-(envd.soc-soc0))
                ss.append(s)
                j=[f+s*pb*envd.dt/3600 for (_,_,pb,f) in cs]; m,u,_,_=cs[int(np.argmin(j))]
                a={"mode":m,"split":np.array([u],np.float32)}
            _,_,t,tr,_=envd.step(a); done=t or tr
        S=envd.get_summary(); cs_=S["total_fuel_g"]-K*(S["final_soc"]-soc0*100)
        return cs_,S,np.array(ss)
    r={}
    for mode in ('aecms','dp'):
        f,S,ss=sim(mode); r[mode]=f; r[mode+'_socf']=S['final_soc']; r[mode+'_par']=S['parallel_percent']; r[mode+'_ser']=S['series_percent']
        r[mode+'_s_p10_p50_p90']=tuple(np.round(np.percentile(ss,[10,50,90]),0)) if len(ss) else None
    r.update(cycle=n,Jstar=j0); print(r,flush=True); return r
if __name__=="__main__":
    n=sys.argv[1]; r=run(n); pd.DataFrame([r]).to_csv(f'headroom_{n}.csv',index=False)
