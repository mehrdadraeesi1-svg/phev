"""
Aggregate plant-v3 results.
    python analyze_v3.py      -> results_v3/per_run.csv, summary.csv, all_per_cycle.csv

Charge-sustaining correction (plant-measured, recharge_v13.py):
    fuel_cs = fuel - K_REPL * E_nom * (SOC_f - SOC_0)/100,  K_REPL = 245 g/kWh, E_nom = 17.5 kWh
Savings are reported vs the rule-based controller and vs A-ECMS on the same plant,
same cycle, same SOC0 (positive = the agent uses less fuel).
"""
import os, glob
import pandas as pd

K_REPL, E_NOM = 245.0, 17.5
R = "results_v3"


def cs(df):
    return df["total_fuel_g"] - K_REPL * E_NOM * (df["final_soc"] - df["soc0"]) / 100.0


def main():
    b = pd.read_csv(os.path.join(R, "baselines.csv"))
    b["fuel_cs"] = cs(b)
    base = b.pivot_table(index=["cycle", "soc0"], columns="controller", values="fuel_cs").reset_index()
    base.columns = ["cycle", "soc0", "ecms_cs", "rb_cs"]

    runs = [pd.read_csv(f) for f in glob.glob(os.path.join(R, "*", "summary.csv"))]
    if not runs:
        print("no RL results yet"); return
    d = pd.concat(runs, ignore_index=True)
    d["seed"] = d["run"].str.extract(r"_s(\d+)_").astype(int)
    d["fuel_cs"] = cs(d)
    d = d.merge(base, on=["cycle", "soc0"])
    d["save_vs_rb_pct"] = 100 * (d.rb_cs - d.fuel_cs) / d.rb_cs
    d["save_vs_ecms_pct"] = 100 * (d.ecms_cs - d.fuel_cs) / d.ecms_cs
    d["terminated"] = d["termination_reason"].notna()
    d.to_csv(os.path.join(R, "all_per_cycle.csv"), index=False)

    cols = ["fuel_cs", "rb_cs", "ecms_cs", "final_soc", "mode_switches", "engine_starts",
            "ev_percent", "series_percent", "parallel_percent", "engine_only_percent",
            "n_vetoed", "n_overridden", "terminated"]
    pr = d.groupby(["scenario", "seed", "run", "soc0"])[cols].mean().reset_index()
    pr["save_vs_rb_pct"] = 100 * (pr.rb_cs - pr.fuel_cs) / pr.rb_cs
    pr["save_vs_ecms_pct"] = 100 * (pr.ecms_cs - pr.fuel_cs) / pr.ecms_cs
    pr.to_csv(os.path.join(R, "per_run.csv"), index=False)

    sm = pr.groupby(["scenario", "soc0"]).agg(
        n_seeds=("seed", "nunique"),
        fuel_cs=("fuel_cs", "mean"), fuel_cs_sd=("fuel_cs", "std"),
        save_vs_rb=("save_vs_rb_pct", "mean"), save_vs_rb_sd=("save_vs_rb_pct", "std"),
        save_vs_ecms=("save_vs_ecms_pct", "mean"), save_vs_ecms_sd=("save_vs_ecms_pct", "std"),
        switches=("mode_switches", "mean"), starts=("engine_starts", "mean"),
        par=("parallel_percent", "mean"), eo=("engine_only_percent", "mean"),
        ser=("series_percent", "mean"), ev=("ev_percent", "mean")).round(2)
    sm.to_csv(os.path.join(R, "summary.csv"))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(sm)
    bb = b.groupby(["controller", "soc0"]).fuel_cs.mean().round(1)
    print("\nbaselines (mean CS-corrected fuel over 8 cycles):\n", bb)


if __name__ == "__main__":
    main()
