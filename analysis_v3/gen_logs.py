import sys, os, glob; sys.path.insert(0,'.')
import pandas as pd, PHEVEnv_v13 as E, ecms_v13 as A
from stable_baselines3 import SAC
os.makedirs('logs_v3',exist_ok=True)
cyc=E.load_cycles(list(E.CYCLE_FILES))
b=pd.read_csv('results_v3/baselines.csv')
out=[]
for n,cd in cyc.items():
    env=E.PHEVMultiModeEnv([cd]); E.run_policy(env,E.rb_action,0.5); L=env.get_logs(); L['ctrl']='RB'; L['seed']=0; L['cycle']=n; out.append(L)
    env=E.PHEVMultiModeEnv([cd]); c=A.precompute(env,cd)
    A.simulate(env,cd,c,258.0,0.5); L=env.get_logs(); L['ctrl']='A-ECMS'; L['seed']=0; L['cycle']=n; out.append(L)
for r in sorted(glob.glob('trained_models_v3_2M/*/best_model.zip')):
    run=r.split('/')[1]; sc=run.split('_s')[0]; seed=int(run.split('_s')[1][:2])
    m=SAC.load(r,device='cpu')
    for n,cd in cyc.items():
        env=E.make_env(sc,[cd],deterministic=True); o,_=env.reset(options={'soc_init':0.5}); d=False
        while not d:
            a,_=m.predict(o,deterministic=True); o,_,t,tr,_=env.step(a); d=t or tr
        L=E._inner(env).get_logs(); L['ctrl']=sc+' (2M)'; L['seed']=seed; L['cycle']=n; out.append(L)
    print(run,flush=True)
pd.concat(out).to_pickle('logs_v3/logs_soc50.pkl')
print('done')
