"""Plant-based counterfactual check of the critic diagnostic (reviewer 2, comment 10):
from states visited by the deterministic G/MA (2M) policies where a coupled mode is admissible,
compare over a 60-s horizon (i) the policy itself with (ii) a forced PARALLEL request at the best
of five load fractions (then back to the policy), both simulated on the plant with dwell and
supervisor; cost = fuel + k E_nom (SOC_start - SOC_end)."""
import sys, glob, copy, numbers, numpy as np, pandas as pd, torch, PHEVEnv_v13 as E
from stable_baselines3 import SAC
torch.set_num_threads(1)
K = 245 * 17.5 / 100; H = 60; EVERY = 60
def snap(objs): return [{k: copy.copy(v) for k, v in o.__dict__.items()
                          if isinstance(v, (numbers.Number, E.DriveMode, dict, type(None), str)) and k not in ("_maps",)} for o in objs]
def restore(objs, s):
    for o, d in zip(objs, s): o.__dict__.update({k: copy.copy(v) for k, v in d.items()})
rows = []
for r in sorted(glob.glob("trained_models_v3_2M/GH-SAC-MA_*/best_model.zip")):
    seed = int(r.split("_s")[1][:2]); m = SAC.load(r, device="cpu")
    for cyc in sys.argv[1:]:
        cd = E.load_cycles([cyc])[cyc]
        env = E.make_env("GH-SAC-MA", [cd], deterministic=True); ie = E._inner(env); objs = (ie, ie._ice, ie._batt)
        o, _ = env.reset(options={"soc_init": 0.5}); t = 0; done = False
        while not done and t < len(cd) - H - 2:
            _, v, P = cd[t]
            if t % EVERY == 0 and v * 3.6 >= 55 and P >= 5 and ie.current_mode != E.DriveMode.PARALLEL:
                s0 = snap(objs); ie._log_on = False
                def run(first):
                    restore(objs, s0); ob = ie._get_obs(); soc_a = ie.soc; fuel = 0.0; mode_used = None
                    for h in range(H):
                        if first is not None and h < 10:
                            a = first
                        else:
                            act, _ = m.predict(ob, deterministic=True); l = act[:4]
                            a = {"mode": int(np.argmax(l)), "split": np.array([np.clip((act[4] + 3) / 6, 0, 1)], np.float32)}
                        n0 = len(ie._log); ob, _, te, tr, _ = ie.step(a); fuel += ie._log[-1]["fuel_g"]
                        if h == 0: mode_used = ie.current_mode.name
                        if te: break
                    return fuel + K * 100 * (soc_a - ie.soc), mode_used
                c_pol, m_pol = run(None)
                alls = [run({"mode": 2, "split": np.array([u], np.float32)}) + (u,) for u in (0.2, 0.4, 0.6, 0.8, 1.0)]
                best = min(alls, key=lambda x: x[0])
                rec = dict(seed=seed, cycle=cyc, t=t, v=v * 3.6, P=P, pol_mode=m_pol, cost_pol=c_pol, cost_par=best[0], par_applied=best[1], u=best[2])
                for c_, mu_, u_ in alls: rec[f"cost_u{u_}"] = c_
                rows.append(rec)
                restore(objs, s0)
            a, _ = m.predict(o, deterministic=True); o, _, te, tr, _ = env.step(a); done = te or tr; t += 1
        print(seed, cyc, len(rows), flush=True)
df = pd.DataFrame(rows); df.to_csv("cf_parallel_all_u.csv", index=False)
df["gain"] = df.cost_pol - df.cost_par
print(df.groupby("cycle").apply(lambda g: pd.Series(dict(n=len(g), share_par_better=(g.gain > 0).mean(), mean_gain=g.gain.mean(), med_gain=g.gain.median()))))
print(df.par_applied.value_counts())
