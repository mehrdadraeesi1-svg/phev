"""
Test trained plant-v3 agents (deterministic policy) on the 8 cycles at SOC0 = 50 % and 70 %.

    python test_v3.py                      # every run in trained_models_v3 without results yet
    python test_v3.py --run trained_models_v3/GH-SAC_s42_2026...   # one run

Per run -> results_v3/<run_name>/summary.csv  (one row per cycle x SOC0)
                                 logs/<cycle>_soc<50|70>.csv.gz  (per-step log)
Baselines (RB, A-ECMS) are produced by baselines_v3.py.
"""
import os, sys, glob, argparse, warnings
os.environ.setdefault("PYTHONUTF8", "1")
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from stable_baselines3 import SAC
import PHEVEnv_v13 as E
import method_v4 as M


def config_of(run_dir):
    c = {}
    for line in open(os.path.join(run_dir, "config.txt"), encoding="utf-8"):
        if ":" in line:
            k, v = line.split(":", 1); c[k.strip()] = v.strip()
    return c


def scenario_of(run_dir):
    return config_of(run_dir)["scenario"]


def test_run(run_dir, data_dir=E.DATA_DIR, out_root="results_v4", model_file="best_model.zip"):
    sc = scenario_of(run_dir)
    name = os.path.basename(os.path.normpath(run_dir))
    out = os.path.join(out_root, name)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    mf = os.path.join(run_dir, model_file)
    if not os.path.exists(mf):
        mf = os.path.join(run_dir, "final_model.zip")
    model = SAC.load(mf, device="cpu")
    cyc = E.load_cycles(list(E.CYCLE_FILES), data_dir)
    rows = []
    for soc0 in (0.5, 0.7):
        for n, cd in cyc.items():
            cfg = config_of(run_dir)
            env = M.make_env_v4(sc, [cd], deterministic=True, center_eta=0.0,
                                dwell_obs=bool(int(cfg.get("dwell_obs", 1))), w_veto=float(cfg.get("w_veto", 0.05)))
            obs, _ = env.reset(options={"soc_init": soc0})
            done = False
            while not done:
                a, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(a)
                done = term or trunc
            inner = E._inner(env)
            s = inner.get_summary()
            inner.get_logs().to_csv(os.path.join(out, "logs", f"{n}_soc{int(soc0*100)}.csv.gz"), index=False)
            rows.append(dict(run=name, scenario=sc, cycle=n, soc0=soc0 * 100, **s))
            print(f"  {sc:11s} {n:14s} SOC0={soc0*100:.0f} fuel={s['total_fuel_g']:7.1f} "
                  f"SOC_f={s['final_soc']:5.1f} sw={s['mode_switches']:4d} PAR={s['parallel_percent']:5.1f}% "
                  f"SER={s['series_percent']:5.1f}% EO={s['engine_only_percent']:4.1f}% "
                  f"{s['termination_reason'] or ''}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "summary.csv"), index=False)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--models", default="trained_models_v4")
    ap.add_argument("--out", default="results_v4")
    ap.add_argument("--data_dir", default=E.DATA_DIR)
    ap.add_argument("--model_file", default="best_model.zip", help="best_model.zip or final_model.zip")
    a = ap.parse_args()
    runs = [a.run] if a.run else sorted(d for d in glob.glob(os.path.join(a.models, "*"))
                                        if os.path.exists(os.path.join(d, "best_model.zip"))
                                        or os.path.exists(os.path.join(d, "final_model.zip")))
    for r in runs:
        if not a.run and os.path.exists(os.path.join(a.out, os.path.basename(r), "summary.csv")):
            print("skip (done):", r); continue
        print("=== testing", r, flush=True)
        test_run(r, a.data_dir, a.out, model_file=a.model_file)
