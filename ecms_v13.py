"""
A-ECMS baseline on the unified plant-v3 environment (PHEVEnv_v13).

At every step the controller evaluates the candidate actions
    EV, SERIES, ENGINE_ONLY, PARALLEL(u = 0.2, 0.4, 0.6, 0.8, 1.0)
with the environment's own power-flow model (warm-engine fuel map) and applies
    argmin  m_f(a) + s(SOC) * P_chem(a) * dt / 3600  (P_chem = I V_oc, internal battery power),   s(SOC) = s0 * (1 - K_SOC (SOC - SOC_0))
through env.step(), i.e. with the SAME supervisor, minimum-dwell constraint and
physics as the RL agents and the rule-based controller.  While the dwell
constraint is active the candidate set is restricted to the current mode (only
its operating point can change).  s0 is bisected per cycle so that the final SOC
returns to SOC_0 (charge-sustaining with cycle knowledge -> an optimistic
reference, as usual for A-ECMS tuned per cycle).

usage:  python ecms_v13.py [--soc0 0.5] [--cycles nedc us06 ...] [--out ecms_v13_soc50.csv]
"""
import argparse, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import PHEVEnv_v13 as E

K_SOC = 1.0
SPLITS = (0.2, 0.4, 0.6, 0.8, 1.0)
SER_SPLITS = (0.0, 0.25, 0.5, 0.75, 1.0)   # only used with a variable (OOL) SERIES point


def precompute(env, cd):
    """Candidate list per step: (mode, u, P_chem_kW, fuel_g).  Battery power is
    evaluated as internal (chemical) power I*V_oc so that I^2 R losses are priced."""
    env.soc = 0.5
    b = env._batt
    b.soc, b.T_batt = 0.5, 30.0
    voc = b.get_ocv(0.5, 30.0)
    out = []
    for k in range(len(cd) - 1):
        _, v, P = cd[k]
        if P < -0.5:
            out.append(None); continue
        cands = [(0, 0.5)] + ([(1, u) for u in SER_SPLITS] if getattr(env, "series_ool", False) else [(1, 0.5)])
        if v * 3.6 >= env.veh.parallel_min_speed_kmh and P >= 3.0:
            cands += [(3, 0.5)] + [(2, u) for u in SPLITS]
        lst = []
        for m, u in cands:
            P_ice, rpm, im, _, P_batt, _ = env.power_flow(E.DriveMode(m), u, v, P)
            f = env.fuel_rate_warm(P_ice, rpm) * env.dt if im != "off" else 0.0
            lst.append((m, u, b.power_to_current(P_batt) * voc / 1000.0, f))
        out.append(lst)
    return out


def simulate(env, cd, cands, s0, soc0):
    env.reset(options={"soc_init": soc0})
    done = False
    while not done:
        k = env.steps
        cs = cands[k] if k < len(cands) else None
        if cs is None:
            a = {"mode": 0, "split": np.array([0.5], np.float32)}
        else:
            if env.dwell < env.min_dwell and env.current_mode != E.DriveMode.REGEN:
                same = [c for c in cs if c[0] == env.current_mode.value]
                cs = same or cs
            s = s0 * (1.0 - K_SOC * (env.soc - soc0))
            j = [f + s * pb * env.dt / 3600.0 for (_, _, pb, f) in cs]
            m, u, _, _ = cs[int(np.argmin(j))]
            a = {"mode": m, "split": np.array([u], np.float32)}
        _, _, term, trunc, _ = env.step(a)
        done = term or trunc
    return env.get_summary()


def tune(cd, soc0=0.5, iters=14, min_dwell=E.MIN_DWELL, series_ool=False):
    env = E.PHEVMultiModeEnv([cd], min_dwell=min_dwell, series_ool=series_ool)
    cands = precompute(env, cd)
    lo, hi, best = 50.0, 1500.0, None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = simulate(env, cd, cands, mid, soc0)
        if s["termination_reason"]:
            lo = mid; continue
        err = abs(s["final_soc"] - soc0 * 100)
        if best is None or err < best[0]:
            best = (err, mid, s)
        if s["final_soc"] < soc0 * 100:
            lo = mid
        else:
            hi = mid
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--soc0", type=float, default=0.5)
    ap.add_argument("--cycles", nargs="*", default=list(E.CYCLE_FILES))
    ap.add_argument("--min_dwell", type=int, default=E.MIN_DWELL)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for n in a.cycles:
        cd = E.load_cycles([n])[n]
        err, s0, s = tune(cd, a.soc0, min_dwell=a.min_dwell)
        rows.append(dict(cycle=n, s0=s0, **{k: v for k, v in s.items() if k != "termination_reason"}))
        print(f"{n:14s} s0={s0:7.1f} fuel={s['total_fuel_g']:7.1f} SOC_f={s['final_soc']:5.1f} "
              f"sw={s['mode_switches']:4d} starts={s['engine_starts']:3d} PAR={s['parallel_percent']:5.1f}% "
              f"SER={s['series_percent']:5.1f}% EO={s['engine_only_percent']:4.1f}%", flush=True)
    out = a.out or f"ecms_v13_soc{int(a.soc0*100)}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("saved", out)
