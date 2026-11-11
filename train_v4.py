"""
v4 (proposed method): plant-v3 + dwell-aware observation, refused-request penalty and
optional reward centering.  python train_v4.py --scenario GH-SAC-MA --gamma 0.999 --center 1e-3 --tag rc999
(original v3 docstring follows)
Training on the unified plant-v3 environment (PHEVEnv_v13).

    python train_v3.py --scenario GH-SAC --seed 42 [--timesteps 500000]

Scenarios: CS-SAC, DH-SAC, GH-SAC, GH-SAC-Eco, DH-SAC-EFC, GH-SAC-SR
(see PHEVEnv_v13.SCENARIOS).  SAC hyper-parameters are identical for all.
Output: trained_models_v3/<scenario>_s<seed>_<timestamp>/ (best_model.zip,
checkpoints every 50k steps, training_log.csv, config.txt)
"""
import os, sys, argparse, warnings
from datetime import datetime
os.environ.setdefault("PYTHONUTF8", "1")
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import PHEVEnv_v13 as E
import method_v4 as M


class LogCallback(BaseCallback):
    def __init__(self, log_dir, check_freq=5000):
        super().__init__(0)
        self.log_dir, self.check_freq = log_dir, check_freq
        self.rows, self._r = [], 0.0

    def _on_step(self):
        i0 = self.locals["infos"][0]
        self._r += float(i0.get("r_raw", self.locals["rewards"][0]))
        if self.locals["dones"][0]:
            i = self.locals["infos"][0]
            self.rows.append(dict(step=self.num_timesteps, reward=self._r, fuel_g=i.get("total_fuel_g", 0),
                                  final_soc=i.get("soc", 0) * 100, switches=i.get("mode_switches", 0),
                                  ev_pct=i.get("ev_percent", 0), ser_pct=i.get("series_percent", 0),
                                  par_pct=i.get("parallel_percent", 0), eo_pct=i.get("engine_only_percent", 0)))
            self._r = 0.0
        if self.n_calls % self.check_freq == 0 and len(self.rows) >= 3:
            d = pd.DataFrame(self.rows[-30:])
            print(f"[{self.num_timesteps:>8,}] R={d.reward.mean():+8.1f} fuel={d.fuel_g.mean():7.1f}g "
                  f"sw={d.switches.mean():5.0f} EV={d.ev_pct.mean():5.1f}% SER={d.ser_pct.mean():5.1f}% "
                  f"PAR={d.par_pct.mean():5.1f}% EO={d.eo_pct.mean():4.1f}%", flush=True)
        return True

    def _on_training_end(self):
        pd.DataFrame(self.rows).to_csv(os.path.join(self.log_dir, "training_log.csv"), index=False)


class Curriculum(BaseCallback):
    def __init__(self, env, all_cycles, at_step):
        super().__init__(0)
        self.env, self.all_cycles, self.at, self.done = env, all_cycles, at_step, False

    def _on_step(self):
        if not self.done and self.num_timesteps >= self.at:
            E._inner(self.env).cycles = self.all_cycles
            print(f"[curriculum] step {self.num_timesteps}: all {len(self.all_cycles)} training cycles", flush=True)
            self.done = True
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=list(E.SCENARIOS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--data_dir", default=E.DATA_DIR)
    ap.add_argument("--save_dir", default="trained_models_v4")
    ap.add_argument("--min_dwell", type=int, default=E.MIN_DWELL)
    ap.add_argument("--gamma", type=float, default=0.99, help="SAC discount factor")
    ap.add_argument("--tag", default="", help="suffix of the run name, e.g. g999")
    ap.add_argument("--dwell_obs", type=int, default=1, help="append dwell timer to observation")
    ap.add_argument("--w_veto", type=float, default=0.05, help="penalty per refused mode request")
    ap.add_argument("--center", type=float, default=0.0, help="reward-centering rate eta (0 = off)")
    ap.add_argument("--threads", type=int, default=1, help="torch CPU threads (1 is fastest when lanes run in parallel)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(a.save_dir, f"{a.scenario}{('_' + a.tag) if a.tag else ''}_s{a.seed}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    cyc = E.load_cycles(E.TRAIN_CYCLES, a.data_dir)
    all_list = [cyc[n] for n in E.TRAIN_CYCLES]
    use_cur = E.SCENARIOS[a.scenario][3]
    init = [cyc["nedc"], cyc["us06"]] if use_cur else all_list

    kw = dict(dwell_obs=bool(a.dwell_obs), w_veto=a.w_veto)
    train_env = Monitor(M.make_env_v4(a.scenario, init, deterministic=False, min_dwell=a.min_dwell, seed=a.seed,
                                      center_eta=a.center, **kw), run_dir)
    # evaluation (checkpoint selection) always uses the un-centred reward
    eval_env = Monitor(M.make_env_v4(a.scenario, [cyc["nedc"]], deterministic=True, min_dwell=a.min_dwell,
                                     seed=a.seed + 1, center_eta=0.0, **kw))

    model = SAC("MlpPolicy", train_env, learning_rate=3e-4, buffer_size=500_000, batch_size=256,
                tau=0.005, gamma=a.gamma, train_freq=1, gradient_steps=1, ent_coef="auto_0.3",
                target_entropy="auto", policy_kwargs=dict(net_arch=[256, 256]), verbose=0, seed=a.seed)
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        act, energy, wsd, cur = E.SCENARIOS[a.scenario]
        for k, v in dict(scenario=a.scenario, action=act, reward=energy, w_soc_delta=wsd or 200.0,
                         curriculum=cur, seed=a.seed, gamma=a.gamma,
                         dwell_obs=a.dwell_obs, w_veto=a.w_veto, center=a.center, timesteps=a.timesteps, min_dwell=a.min_dwell,
                         plant=E.PLANT, data_dir=a.data_dir, timestamp=ts,
                         obs_dim=train_env.observation_space.shape[0],
                         act_dim=train_env.action_space.shape[0]).items():
            f.write(f"{k}: {v}\n")
    print(f"=== {a.scenario} seed {a.seed} | plant {E.PLANT} | {a.timesteps:,} steps -> {run_dir}", flush=True)

    cbs = [LogCallback(run_dir),
           CheckpointCallback(save_freq=50_000, save_path=run_dir, name_prefix="sac_ckpt"),
           EvalCallback(eval_env, best_model_save_path=run_dir, log_path=run_dir, eval_freq=20_000,
                        n_eval_episodes=3, deterministic=True, verbose=0)]
    if use_cur:
        cbs.append(Curriculum(train_env, all_list, int(0.3 * a.timesteps)))
    t0 = datetime.now()
    model.learn(total_timesteps=a.timesteps, callback=CallbackList(cbs), progress_bar=False)
    model.save(os.path.join(run_dir, "final_model"))
    print(f"done in {(datetime.now() - t0).total_seconds() / 60:.1f} min -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
