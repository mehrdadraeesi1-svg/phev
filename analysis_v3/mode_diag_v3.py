"""Mode-exploration and critic diagnostics on plant-v3 checkpoints (seed 42, one run per factorial cell).
For states visited by the deterministic policy on FTP-75 and US06 (SOC0 = 50 %):
  H_mode     entropy of the applied mode under the TRAINING behaviour (actor sample -> argmax or Gumbel-max)
  p_par      probability that PARALLEL is applied under that behaviour
  dQ_force   Q(s, a with l_PAR raised above the current max) - Q(s, mean action)
  ecms_par   whether the plant-model ECMS cost (s = 258 g/kWh) prefers a PARALLEL candidate in that state
Metrics are also reported on 'coupled-admissible' states (v >= 50 km/h, P_dem >= 3 kW)."""
import sys, glob, re; sys.path.insert(0,'.')
import numpy as np, pandas as pd, torch
from stable_baselines3 import SAC
import PHEVEnv_v13 as E
PAR=2; NS=256; rng=np.random.default_rng(0)
CELLS={"DH-SAC_s42_20260926_183603":("argmax","sep"),"DH-SAC-EFC_s42_20260926_184222":("argmax","efc"),
       "DH-SAC-MA_s42_20260927_113633":("argmax","aligned"),"GH-SAC-SR_s42_20260926_184250":("gumbel","sep"),
       "GH-SAC_s42_20260926_183605":("gumbel","efc"),"GH-SAC-MA_s42_20260927_113642":("gumbel","aligned")}
cyc=E.load_cycles(["ftp75","us06"])
probe=E.PHEVMultiModeEnv([cyc["ftp75"]]); bt=probe._batt; bt.soc,bt.T_batt=0.5,30.0; voc=bt.get_ocv(0.5,30.0)
def ecms_prefers_par(v,P,soc):
    if v*3.6<50 or P<3: return False
    probe.soc=0.5; best=None
    for m,u in [(0,.5),(1,.5),(3,.5)]+[(2,x) for x in (0.2,0.4,0.6,0.8,1.0)]:
        Pi,rpm,im,_,Pb,_=probe.power_flow(E.DriveMode(m),u,v,P)
        f=probe.fuel_rate_warm(Pi,rpm) if im!="off" else 0.0
        j=f+258.0*(1-(soc-0.5))*bt.power_to_current(Pb)*voc/1000/3600
        if best is None or j<best[0]: best=(j,m)
    return best[1]==2
def rollout(model,sc):
    S,info=[],[]
    for n,cd in cyc.items():
        env=E.make_env(sc,[cd],deterministic=True); o,_=env.reset(options={"soc_init":0.5}); inner=E._inner(env); d=False
        while not d:
            _,v,P=inner.cycle_data[inner.steps]
            S.append(o.copy()); info.append((v,P,inner.soc))
            a,_=model.predict(o,deterministic=True); o,_,t,tr,_=env.step(a); d=t or tr
    return np.array(S,np.float32),np.array(info)
def diag(model,S,lin):
    pol=model.policy; ot=torch.as_tensor(S)
    with torch.no_grad():
        mean=pol.actor(ot,deterministic=True)
        smp=torch.stack([pol.actor(ot,deterministic=False) for _ in range(NS)],1)
    l=3.0*smp[...,:4].numpy()
    if lin=="gumbel":
        l=l-np.log(-np.log(rng.uniform(1e-10,1-1e-10,size=l.shape)))
    modes=l.argmax(-1); pm=np.stack([(modes==k).mean(1) for k in range(4)],1)
    H=-(pm*np.log(pm+1e-12)).sum(1)
    with torch.no_grad():
        q0=torch.min(*pol.critic(ot,mean)).squeeze(1)
        best=None
        for u in (0.2,0.4,0.6,0.8,1.0):   # best engine load fraction for the forced PARALLEL action
            af=mean.clone(); top=af[:,:4].max(1).values
            af[:,PAR]=torch.clamp(top+0.5/3,max=1.0); af[:,PAR]=torch.where(af[:,PAR]<=top,top+1e-3,af[:,PAR])
            af[:,4]=(6*u-3)/3.0
            q=torch.min(*pol.critic(ot,af)).squeeze(1)
            best=q if best is None else torch.maximum(best,q)
        dq=(best-q0).numpy()
    return H,pm[:,PAR],dq,(3*mean[:,:4].numpy()).argmax(1)
rows=[]
for run,(lin,rew) in CELLS.items():
    sc=run.split("_s")[0]
    ecms=None
    for ck in ["sac_ckpt_150000_steps","sac_ckpt_300000_steps","sac_ckpt_500000_steps"]:
        m=SAC.load(f"trained_models_v3/{run}/{ck}.zip",device="cpu")
        S,inf=rollout(m,sc)
        H,pp,dq,greedy=diag(m,S,lin)
        adm=(inf[:,0]*3.6>=50)&(inf[:,1]>=3)
        if ecms is None or len(ecms)!=len(S):
            ecms=np.array([ecms_prefers_par(v,P,s) for v,P,s in inf])
        step=int(re.findall(r"(\d+)_steps",ck)[0])
        r=dict(scenario=sc,action=lin,reward=rew,step=step,n=len(S),n_adm=int(adm.sum()),
               H_all=H.mean(),H_adm=H[adm].mean(),p_par_adm=pp[adm].mean(),greedy_par_adm=(greedy[adm]==PAR).mean(),
               dQ_adm=dq[adm].mean(),frac_dQpos_adm=(dq[adm]>0).mean(),
               ecms_par_adm=ecms[adm].mean(),
               dQpos_where_ecms_par=(dq[adm&ecms]>0).mean() if (adm&ecms).any() else np.nan)
        rows.append(r); print({k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items()},flush=True)
pd.DataFrame(rows).to_csv("mode_diag_v3.csv",index=False)
