"""DP grid-convergence and terminal-SOC check (reviewer 2, comment 6)."""
import sys, numpy as np, pandas as pd, PHEVEnv_v13 as E, dp_v3 as D
def forward(grid, J, soc0):
    tabs, bt = D.LAST["tabs"], D.LAST["bt"]; soc = soc0; fuel = 0.0
    for k, per in enumerate(tabs):
        i = int(np.clip(np.searchsorted(grid, soc), 0, grid.size - 1))
        voc = np.interp(soc, grid, bt.voc); R = np.interp(soc, grid, bt.R); best = (np.inf, 0, 0)
        for f, pb in per:
            Pw = pb[i] * 1000.0; disc = voc ** 2 - 4 * R * Pw
            I = (voc - np.sqrt(disc)) / (2 * R) if disc >= 0 else Pw / voc
            sn = soc - np.clip(I, -bt.Imax, bt.Imax) / bt.C
            c = f[i] + np.interp(sn, grid, J[k + 1])
            if c < best[0]: best = (c, f[i], sn)
        fuel += best[1]; soc = best[2]
    return fuel, soc
rows = []
for cyc in sys.argv[1:]:
    cd = E.load_cycles([cyc])[cyc]
    for ds, du, pen in [(5e-4, 0.1, 5), (1e-3, 0.1, 5), (2.5e-4, 0.1, 5), (5e-4, 0.05, 5), (5e-4, 0.1, 20), (5e-4, 0.1, 1.5)]:
        D.DS, D.PENF = ds, pen; D.SPLITS = np.round(np.arange(0, 1 + 1e-9, du), 3)
        env = E.PHEVMultiModeEnv([cd], min_dwell=0); env.reset(options={"soc_init": 0.5})
        grid, J, j0, _ = D.solve(env, cd, 0.5); fuel, socT = forward(grid, J, 0.5)
        cs = fuel - D.K_G_PER_PCT * (socT * 100 - 50)
        rows.append(dict(cycle=cyc, dSOC=ds, du=du, pen=pen, J0=j0, fuel=fuel, socT=socT * 100, cs=cs))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"dp_conv_{'_'.join(sys.argv[1:])}.csv", index=False)
