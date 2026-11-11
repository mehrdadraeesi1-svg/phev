"""
Dynamic-programming benchmark on the plant-v3 environment (PHEVEnv_v13).

State  : SOC (grid 0.10..0.90, dSOC = 5e-4, linear interpolation of cost-to-go)
Control: requested mode  EV, SERIES, ENGINE_ONLY, PARALLEL(u = 0, 0.1, ..., 1.0)
         mapped through the environment's own supervisor (SOC-dependent), so the
         admissible set is identical to that of every other controller.
Stage  : warm-engine fuel [g] from env.fuel_rate_warm (same map as the plant)
Battery: dSOC = -I dt / (C 3600), I from P = Voc I - R I^2 with the plant's
         OCV/R maps at a fixed pack temperature (T_DP, default 27 C).
Terminal: phi(SOC_N) = K (SOC0 - SOC_N)  with K = k E_nom, k = 245 g/kWh, i.e. DP
         minimises exactly the charge-sustaining-corrected fuel used as the metric.
Not modelled in the DP: minimum-dwell constraint, cold-engine penalty, engine
thermal state, battery temperature  ->  a lower bound.

The DP cost-to-go is then used as a closed-loop policy inside the real environment
(argmin fuel + J_{k+1}(SOC_pred) at the measured SOC), with min_dwell = 0 and
min_dwell = 10, so every reported number is also plant-verified.

usage: python dp_v3.py --cycles nedc us06 --soc0 0.5
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import PHEVEnv_v13 as E

K_G_PER_PCT = 245 * 17.5 / 100          # g per % SOC
SPLITS = np.round(np.linspace(0, 1, 11), 2)
S_LO, S_HI, DS = 0.10, 0.90, 5e-4
T_DP = 27.0
TERMINAL = "cs"
PENF = 5.0
LAST = {}
OOL = False


def req_candidates(v, P, env):
    c = [(0, 0.5)] + ([(1, float(u)) for u in SPLITS] if getattr(env, "series_ool", False) else [(1, 0.5)])
    if v * 3.6 >= env.veh.parallel_min_speed_kmh and P >= 3.0:
        c += [(3, 0.5)] + [(2, float(u)) for u in SPLITS]
    return c


class BattTable:
    def __init__(self, env, grid, T=T_DP):
        b = env._batt
        self.voc = np.array([b.get_ocv(s, T) for s in grid])
        self.R = np.array([b.get_resistance(s, T) for s in grid])
        self.Imax = b.I_max
        self.C = b.C_pack * 3600.0

    def dsoc(self, P_kw):
        Pw = P_kw * 1000.0
        disc = self.voc ** 2 - 4 * self.R * Pw
        I = np.where(disc >= 0, (self.voc - np.sqrt(np.maximum(disc, 0))) / (2 * self.R),
                     Pw / self.voc)
        I = np.clip(I, -self.Imax, self.Imax)
        return -I / self.C


def stage_tables(env, cd, grid, bt):
    """For every step: list of (fuel_g [G], dsoc [G]) arrays, one per requested candidate,
    with the supervisor applied at each SOC grid point (piecewise constant in SOC)."""
    # SOC breakpoints of the supervisor / power_flow
    h = env.health
    bps = sorted({0.35, h.SOC_min, h.SOC_min_emergency + 0.05, h.SOC_max})
    edges = [S_LO] + bps + [S_HI + 1e-9]
    reps = [0.5 * (edges[i] + min(edges[i + 1], S_HI)) for i in range(len(edges) - 1)]
    reg_idx = np.searchsorted(np.array(bps), grid, side="right")   # region per grid pt
    # careful: power_flow uses soc >= SOC_max, supervisor soc > SOC_max; the grid never hits exactly 0.85
    tabs = []
    for k in range(len(cd) - 1):
        _, v, P = cd[k]
        cands = req_candidates(v, P, env) if P >= -0.5 else [(0, 0.5)]
        per = []
        for (m, u) in cands:
            f_r, pb_r, fm_r = [], [], []
            for s in reps:
                env.soc = s
                fm, _ = env._supervisor(E.DriveMode(m), v, P)
                P_ice, rpm, im, _, P_batt, _ = env.power_flow(fm, u, v, P)
                f_r.append(env.fuel_rate_warm(P_ice, rpm) if im != "off" else 0.0)
                pb_r.append(P_batt); fm_r.append(fm.value)
            f = np.array(f_r)[reg_idx]
            pb = np.array(pb_r)[reg_idx]
            per.append((f, pb))
        tabs.append(per)
    return tabs


def solve(env, cd, soc0):
    grid = np.arange(S_LO, S_HI + DS / 2, DS)
    bt = BattTable(env, grid)
    t0 = time.time()
    tabs = stage_tables(env, cd, grid, bt)
    LAST.update(tabs=tabs, bt=bt)
    t1 = time.time()
    N = len(tabs)
    J = np.empty((N + 1, grid.size), np.float32)
    if TERMINAL == "linear":
        J[N] = K_G_PER_PCT * (soc0 - grid) * 100.0
    else:   # charge-sustaining: final SOC pinned to SOC0 by a stiff two-sided penalty
        J[N] = PENF * K_G_PER_PCT * np.abs(soc0 - grid) * 100.0
    INF = 1e9
    for k in range(N - 1, -1, -1):
        best = np.full(grid.size, INF)
        for f, pb in tabs[k]:
            Pw = pb * 1000.0
            disc = bt.voc ** 2 - 4 * bt.R * Pw
            I = np.where(disc >= 0, (bt.voc - np.sqrt(np.maximum(disc, 0))) / (2 * bt.R), Pw / bt.voc)
            ds = -np.clip(I, -bt.Imax, bt.Imax) / bt.C
            sn = grid + ds
            ok = (sn >= S_LO) & (sn <= S_HI)
            val = f + np.interp(sn, grid, J[k + 1])
            val = np.where(ok, val, INF)
            best = np.minimum(best, val)
        J[k] = best
    t2 = time.time()
    j0 = float(np.interp(soc0, grid, J[0]))
    return grid, J, j0, (t1 - t0, t2 - t1)


def replay(env_cd, grid, J, soc0, min_dwell):
    """Closed-loop DP policy in the real plant."""
    env = E.PHEVMultiModeEnv([env_cd], min_dwell=min_dwell, series_ool=OOL)
    env.reset(options={"soc_init": soc0})
    b = env._batt
    done = False
    while not done:
        k = env.steps
        _, v, P = env.cycle_data[k]
        if P < -0.5:
            a = {"mode": 0, "split": np.array([0.5], np.float32)}
        else:
            cands = req_candidates(v, P, env)
            if min_dwell > 0 and env.dwell < env.min_dwell and env.current_mode != E.DriveMode.REGEN:
                same = [c for c in cands if c[0] == env.current_mode.value]
                cands = same or cands
            best, a = np.inf, None
            b.soc, b.T_batt = env.soc, env.T_batt
            for (m, u) in cands:
                fm, _ = env._supervisor(E.DriveMode(m), v, P)
                P_ice, rpm, im, _, P_batt, _ = env.power_flow(fm, u, v, P)
                f = env.fuel_rate_warm(P_ice, rpm) if im != "off" else 0.0
                I = b.power_to_current(P_batt)
                sn = env.soc - I / (b.C_pack * (1 - env.qloss_total) * 3600.0)
                c = f + np.interp(sn, grid, J[k + 1]) if S_LO <= sn <= S_HI else np.inf
                if c < best:
                    best, a = c, {"mode": m, "split": np.array([u], np.float32)}
        _, _, term, trunc, _ = env.step(a)
        done = term or trunc
    return env.get_summary()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", nargs="*", default=["nedc", "us06"])
    ap.add_argument("--soc0", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ool", action="store_true", help="variable SERIES point on the optimal operating line")
    ap.add_argument("--terminal", default="cs", choices=["cs", "linear"])
    a = ap.parse_args()
    TERMINAL = a.terminal
    OOL = a.ool
    rows = []
    for n in a.cycles:
        cd = E.load_cycles([n])[n]
        env = E.PHEVMultiModeEnv([cd], min_dwell=0, series_ool=a.ool)
        env.reset(options={"soc_init": a.soc0})
        grid, J, j0, (tp, ts) = solve(env, cd, a.soc0)
        r = dict(cycle=n, soc0=a.soc0 * 100, dp_model_cs=j0, t_pre=tp, t_solve=ts)
        for md in (0, E.MIN_DWELL):
            s = replay(cd, grid, J, a.soc0, md)
            cs = s["total_fuel_g"] - K_G_PER_PCT * (s["final_soc"] - a.soc0 * 100)
            tag = f"dw{md}"
            r.update({f"{tag}_fuel": s["total_fuel_g"], f"{tag}_socf": s["final_soc"], f"{tag}_cs": cs,
                      f"{tag}_ev": s["ev_percent"], f"{tag}_ser": s["series_percent"],
                      f"{tag}_par": s["parallel_percent"], f"{tag}_eo": s["engine_only_percent"],
                      f"{tag}_sw": s["mode_switches"], f"{tag}_starts": s["engine_starts"],
                      f"{tag}_veto": s["n_vetoed"], f"{tag}_rms": s["rms_current_A"],
                      f"{tag}_peakT": s["peak_T_batt"]})
        rows.append(r)
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    out = a.out or f"dp_v3{'_ool' if a.ool else ''}_soc{int(a.soc0 * 100)}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("saved", out)
