"""Battery-thermal sensitivity (reviewer minor comment): evaluate trained agents and baselines
without retraining under hotter ambient and/or without active pack cooling.
usage: python thermal_sens.py <scen> ...   scen in base, nocool, hot, hot_nocool"""
import sys, os, glob, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
import PHEVEnv_v13 as E, ecms_v13 as A
from stable_baselines3 import SAC
torch.set_num_threads(1)
SCEN = {"base": (25.0, 0.02), "nocool": (25.0, 0.2), "hot": (40.0, 0.02), "hot_nocool": (40.0, 0.2)}
K = 245 * 17.5 / 100

def inner(cd, tamb, rth, md=E.MIN_DWELL, rc=None):
    # thermal termination disabled: the sensitivity reports temperatures instead of stopping the run
    env = E.PHEVMultiModeEnv([cd], thermal_params=E.ThermalParams(T_ambient=tamb, T_batt_emergency=500.0), min_dwell=md,
                             reward_config=rc)
    env._batt.R_thermal = rth
    return env

def row(env, ctrl, seed, cyc, sc):
    s = env.get_summary(); L = env.get_logs()
    return dict(scen=sc, ctrl=ctrl, seed=seed, cycle=cyc, fuel=s["total_fuel_g"], socf=s["final_soc"],
                steps=s["total_steps"], term=str(s["termination_reason"]), cs=s["total_fuel_g"] - K * (s["final_soc"] - 50.0), peakT=s["peak_T_batt"],
                meanT=float(L["T_battery"].mean()), t_above45=float((L["T_battery"] > 45).mean() * 100),
                rms=s["rms_current_A"], qloss=s["qloss_total"], par=s["parallel_percent"], ser=s["series_percent"])

cyc = E.load_cycles(list(E.CYCLE_FILES))
models = sorted(glob.glob("trained_models_v3_2M/GH-SAC-MA_*/best_model.zip"))
for sc in sys.argv[1:]:
    tamb, rth = SCEN[sc]; rows = []; t0 = time.time()
    for n, cd in cyc.items():
        env = inner(cd, tamb, rth); E.run_policy(env, E.rb_action, 0.5); rows.append(row(env, "RB", 0, n, sc))
        env = inner(cd, tamb, rth); c = A.precompute(env, cd); A.simulate(env, cd, c, 258.0, 0.5)
        rows.append(row(env, "A-ECMS", 0, n, sc))
    for r in models:
        run = r.split("/")[1]; seed = int(run.split("_s")[1][:2]); m = SAC.load(r, device="cpu")
        for n, cd in cyc.items():
            ie = inner(cd, tamb, rth, rc=E.reward_config_for("GH-SAC-MA"))
            env = E.WRAPPERS["gumbel"](ie, deterministic=True)
            o, _ = env.reset(options={"soc_init": 0.5}); d = False
            while not d:
                a, _ = m.predict(o, deterministic=True); o, _, t, tr, _ = env.step(a); d = t or tr
            rows.append(row(ie, "G/MA (2M)", seed, n, sc))
        print(sc, run, round(time.time() - t0), flush=True)
    pd.DataFrame(rows).to_csv(f"thermal_{sc}.csv", index=False); print("saved", sc, flush=True)
