"""Why does T/WS issue ~1200 dwell-refused requests per cycle?  Logs every step of the
deterministic T/WS policies (3 seeds, 8 cycles, SOC0 50) with the raw mode signal."""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
import PHEVEnv_v13 as E
from stable_baselines3 import SAC
torch.set_num_threads(1)
cyc = E.load_cycles(list(E.CYCLE_FILES))
rows = []
for r in sorted(glob.glob("trained_models_v3/CS-SAC_*/best_model.zip")) or sorted(glob.glob("trained_models_v3/CS-SAC_*/final_model.zip")):
    seed = int(r.split("_s")[1][:2]); m = SAC.load(r, device="cpu")
    for n, cd in cyc.items():
        env = E.make_env("CS-SAC", [cd], deterministic=True); ie = E._inner(env)
        o, _ = env.reset(options={"soc_init": 0.5}); d = False
        while not d:
            a, _ = m.predict(o, deterministic=True)
            cur, dw, nv = ie.current_mode.value, ie.dwell, ie.n_vetoed
            req, _ = E.ThresholdActionWrapper.decode(a)
            o, rwd, t, tr, info = env.step(a); d = t or tr
            c = info.get("reward_components", {})
            rows.append((seed, n, float(a[0]), cur, req, ie.current_mode.value, dw, ie.n_vetoed - nv,
                         rwd, c.get("C_switch", 0.0), c.get("C_emission", 0.0) + c.get("C_energy", 0.0)))
        print(seed, n, flush=True)
df = pd.DataFrame(rows, columns=["seed", "cycle", "sig", "cur", "req", "applied", "dwell", "veto",
                                 "reward", "C_sw", "C_em_en"])
df.to_pickle("logs_v3/tws_diag.pkl"); print("done")
