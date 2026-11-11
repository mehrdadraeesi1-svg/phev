"""A-ECMS driven to the SAME final SOC as a reference controller (per cycle), so that fuel can be
compared without any charge-sustaining correction factor (reviewer 2, comment 7)."""
import sys, numpy as np, pandas as pd, PHEVEnv_v13 as E, ecms_v13 as A
K = 245 * 17.5 / 100
d = pd.read_csv("runs_2M_5seed_ool.csv"); g = d[d.scenario == "GH-SAC-MA"]
b = pd.read_csv("results_v3/baselines.csv"); b = b[b.controller == "RB"]
refs = {"G/MA": g.groupby(["cycle", "soc0"]).agg(fuel=("total_fuel_g", "mean"), socT=("final_soc", "mean")),
        "RB": b.groupby(["cycle", "soc0"]).agg(fuel=("total_fuel_g", "mean"), socT=("final_soc", "mean"))}
rows = []
for soc0 in map(float, sys.argv[1:]):
    for n, cd in E.load_cycles(list(E.CYCLE_FILES)).items():
        env = E.PHEVMultiModeEnv([cd]); c = A.precompute(env, cd)
        for rn, R in refs.items():
            tgt = R.loc[(n, soc0), "socT"]; lo, hi, best = 100.0, 2000.0, None
            for _ in range(16):
                mid = 0.5 * (lo + hi); s = A.simulate(env, cd, c, mid, soc0 / 100)
                if s["termination_reason"]: hi = mid; continue
                err = s["final_soc"] - tgt
                if best is None or abs(err) < abs(best[0]): best = (err, mid, s)
                if err < 0: lo = mid
                else: hi = mid
            err, s0, s = best
            rows.append(dict(ref=rn, cycle=n, soc0=soc0, target=tgt, ecms_socT=s["final_soc"], s0=s0,
                             ecms_fuel=s["total_fuel_g"], ref_fuel=R.loc[(n, soc0), "fuel"],
                             ecms_fuel_adj=s["total_fuel_g"] - K * (s["final_soc"] - tgt)))
            print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"ecms_matched_{'_'.join(sys.argv[1:])}.csv", index=False)
