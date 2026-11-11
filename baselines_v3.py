"""
Rule-based and A-ECMS baselines on plant v3, SOC0 = 50 % and 70 %, all 8 cycles.
    python baselines_v3.py   ->  results_v3/baselines.csv  (+ per-step logs)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import PHEVEnv_v13 as E
import ecms_v13 as A

OUT = "results_v3"
os.makedirs(os.path.join(OUT, "baseline_logs"), exist_ok=True)
rows = []
cyc = E.load_cycles(list(E.CYCLE_FILES))
for soc0 in (0.5, 0.7):
    for n, cd in cyc.items():
        env = E.PHEVMultiModeEnv([cd])
        s = E.run_policy(env, E.rb_action, soc0)
        env.get_logs().to_csv(os.path.join(OUT, "baseline_logs", f"RB_{n}_soc{int(soc0*100)}.csv.gz"), index=False)
        rows.append(dict(controller="RB", cycle=n, soc0=soc0 * 100, s0=None, **s))
        err, s0, s = A.tune(cd, soc0)
        rows.append(dict(controller="A-ECMS", cycle=n, soc0=soc0 * 100, s0=s0, **s))
        print(f"SOC0={soc0*100:.0f} {n:14s} RB fuel={rows[-2]['total_fuel_g']:7.1f} SOC_f={rows[-2]['final_soc']:5.1f} | "
              f"A-ECMS fuel={s['total_fuel_g']:7.1f} SOC_f={s['final_soc']:5.1f} s0={s0:6.1f}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "baselines.csv"), index=False)
print("saved", os.path.join(OUT, "baselines.csv"))
