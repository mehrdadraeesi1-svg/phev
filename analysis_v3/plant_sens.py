"""One-at-a-time plant-parameter sensitivity (reviewer 2, comment 8).
Trained agents are evaluated WITHOUT retraining; A-ECMS uses the perturbed plant model with
s0 scaled by the perturbed replenishment cost; k is re-measured on each perturbed plant.
usage: python plant_sens.py <scen> [...]"""
import sys, os, glob, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
import PHEVEnv_v13 as E, ecms_v13 as A
import PhysicsModels_v10 as PM
from stable_baselines3 import SAC
torch.set_num_threads(1)
SC = {"base": {}, "gen-": dict(generator_efficiency=0.874), "gen+": dict(generator_efficiency=0.966),
      "mot-": dict(motor_efficiency=0.884), "mot+": dict(motor_efficiency=0.976),
      "drv-": dict(driveline_efficiency=0.92), "drv+": dict(driveline_efficiency=0.98),
      "R-": dict(_R=0.7), "R+": dict(_R=1.5), "bsfc_flat": dict(_B=0.5), "bsfc_steep": dict(_B=2.0)}
_R0 = PM.BatteryModelRL.get_resistance; _F0 = PM.ICEModelRL.get_fuel_rate_gs

def patch(p):
    rs, bs = p.get("_R", 1.0), p.get("_B", 1.0)
    PM.BatteryModelRL.get_resistance = lambda self, s, T: rs * _R0(self, s, T)
    def f(self, P, rpm, bmep, T):
        g = _F0(self, P, rpm, bmep, T)
        if g <= 0 or P < 0.1: return g
        b = g * 3600.0 / P
        return g * (1.0 + (bs - 1.0) * (b / 233.0 - 1.0))
    PM.ICEModelRL.get_fuel_rate_gs = f

def vp(p): return E.VehicleParams(**{k: v for k, v in p.items() if not k.startswith("_")})

def k_meas(p):
    cyc = np.array([[t, 0.0, 0.0] for t in range(10)], float); ks = []
    for sf in (0.30, 0.40, 0.45):
        e = E.PHEVMultiModeEnv([cyc], vehicle_params=vp(p)); e.reset(options={"soc_init": sf})
        e.T_eng, e.T_batt = 90.0, 30.0; fuel = 0.0; n = 0
        while e.soc < 0.5 and n < 10000:
            ph = e._apply_physics(E.DriveMode.SERIES, 0.5, 0.0, 0.0)
            e._update_state(ph, E.DriveMode.SERIES, 0.0); fuel += ph["fuel_g"]; n += 1
        ks.append(fuel / ((e.soc - sf) * e.health.E_nominal_kwh))
    return float(np.mean(ks))

cyc = E.load_cycles(list(E.CYCLE_FILES))
models = sorted(glob.glob("trained_models_v3_2M/GH-SAC-MA_*/best_model.zip"))
for sc in sys.argv[1:]:
    p = SC[sc]; patch(p); t0 = time.time(); rows = []
    k = k_meas(p); s0 = 258.0 * k / 245.0
    for n, cd in cyc.items():
        env = E.PHEVMultiModeEnv([cd], vehicle_params=vp(p)); s = E.run_policy(env, E.rb_action, 0.5)
        rows.append(dict(ctrl="RB", seed=0, cycle=n, fuel=s["total_fuel_g"], socf=s["final_soc"], par=s["parallel_percent"], ser=s["series_percent"]))
        env = E.PHEVMultiModeEnv([cd], vehicle_params=vp(p)); c = A.precompute(env, cd); s = A.simulate(env, cd, c, s0, 0.5)
        rows.append(dict(ctrl="A-ECMS", seed=0, cycle=n, fuel=s["total_fuel_g"], socf=s["final_soc"], par=s["parallel_percent"], ser=s["series_percent"]))
    for r in models:
        seed = int(r.split("_s")[1][:2]); m = SAC.load(r, device="cpu")
        for n, cd in cyc.items():
            ie = E.PHEVMultiModeEnv([cd], vehicle_params=vp(p), reward_config=E.reward_config_for("GH-SAC-MA"))
            env = E.WRAPPERS["gumbel"](ie, deterministic=True); o, _ = env.reset(options={"soc_init": 0.5}); d = False
            while not d:
                a, _ = m.predict(o, deterministic=True); o, _, t, tr, _ = env.step(a); d = t or tr
            s = ie.get_summary()
            rows.append(dict(ctrl="G/MA", seed=seed, cycle=n, fuel=s["total_fuel_g"], socf=s["final_soc"], par=s["parallel_percent"], ser=s["series_percent"], term=str(s["termination_reason"])))
    df = pd.DataFrame(rows); df["scen"] = sc; df["k"] = k; df["s0"] = s0
    df.to_csv(f"plant_sens_{sc}.csv", index=False); print("saved", sc, round(k, 1), round(time.time() - t0), flush=True)
