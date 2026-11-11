"""A-ECMS under the variable (OOL) SERIES operating point: tune s0 per cycle, then run the
causal variant with one fixed s0 (median of the tuned values), SOC0 = 50 and 70 %."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, PHEVEnv_v13 as E, ecms_v13 as A
K = 245 * 17.5 / 100
cyc = E.load_cycles(list(E.CYCLE_FILES))
tuned = []
for n, cd in cyc.items():
    err, s0, s = A.tune(cd, 0.5, series_ool=True)
    tuned.append(s0); print(n, "tuned s0", round(s0, 1), round(s["total_fuel_g"], 1), round(s["final_soc"], 2), flush=True)
s0 = float(np.median(tuned)); print("causal s0", s0)
rows = []
for soc0 in (0.5, 0.7):
    for n, cd in cyc.items():
        for ctrl in ("A-ECMS", "RB"):
            env = E.PHEVMultiModeEnv([cd], series_ool=True)
            if ctrl == "RB":
                s = E.run_policy(env, E.rb_action, soc0)
            else:
                c = A.precompute(env, cd); s = A.simulate(env, cd, c, s0, soc0)
            rows.append(dict(controller=ctrl, cycle=n, soc0=soc0 * 100, s0=s0, **{k: v for k, v in s.items() if k != "termination_reason"}))
            print(ctrl, n, soc0, round(s["total_fuel_g"] - K * (s["final_soc"] - soc0 * 100), 1), flush=True)
pd.DataFrame(rows).to_csv("ool_baselines.csv", index=False); print("saved")
