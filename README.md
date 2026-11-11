# Reward-implied energy pricing and discounting bias in deep RL energy management of a multi-mode PHEV

Code, simulation environment, trained agents and result tables for the manuscript

> *Reward-Implied Energy Pricing and Discounting Bias in Deep Reinforcement Learning Energy Management of a Multi-Mode Plug-in Hybrid Electric Vehicle* (under review).

The study compares Soft Actor-Critic (SAC) agents with a rule-based controller, a causal adaptive ECMS and a dynamic-programming bound on one plant model of a series-parallel plug-in hybrid vehicle. All controllers share the same supervisor, 10-s minimum-dwell rule and evaluation.

## Contents

| Path | What it is |
|---|---|
| `PHEVEnv_v13.py` | Gymnasium environment: plant, supervisor, dwell rule, rewards, action mappings, rule-based controller |
| `PhysicsModels_v10.py`, `PhysicsModels_v8.py`, `utiles/` | Engine and battery models and their map loader |
| `train_v3.py`, `test_v3.py` | Training and evaluation of the factorial agents |
| `method_v4.py`, `train_v4.py`, `test_v4.py` | Remedies: dwell-aware observation, refused-request penalty, reward centring |
| `baselines_v3.py`, `ecms_v13.py` | Rule-based and A-ECMS benchmarks |
| `recharge_v13.py` | Measurement of the replenishment cost *k* |
| `analyze_v3.py`, `analysis_v3/` | Aggregation, DP bound (`dp_v3.py`), statistics, sensitivity studies, figure scripts and the resulting CSV tables |
| `data_1s/` | Speed and power traces of seven of the eight driving cycles on a 1-s grid |
| `data_engine/` | Placeholder for the engine maps, which are not distributed (see Data) |
| `trained_models_*/` | Best (NEDC-selected) and final checkpoint, configuration and training log of every run |
| `results_*/` | Per-run `summary.csv` (one row per cycle and initial SOC) and the benchmark tables |
| `batch/` | Windows batch files used for the original runs; copy one to the repository root to run it |

## Scenario names

The code uses its own scenario names. They map to the controller IDs of the paper as follows.

| Paper | Code | Mode mapping | Reward |
|---|---|---|---|
| T/WS | `CS-SAC` | threshold | weighted sum |
| A/Sep | `DH-SAC` | argmax | separate fuel and electricity |
| A/EFC | `DH-SAC-EFC` | argmax | equivalent fuel consumption |
| A/MA | `DH-SAC-MA` | argmax | metric-anchored |
| G/Sep | `GH-SAC-SR` | Gumbel-max | separate fuel and electricity |
| G/EFC | `GH-SAC` | Gumbel-max | equivalent fuel consumption |
| G/MA | `GH-SAC-MA` | Gumbel-max | metric-anchored |
| G/EFC120 | `GH-SAC-Eco` | Gumbel-max | EFC with SOC-change weight 120 |

Run folders: `trained_models_v3` (0.5 M steps, seeds 42-44), `trained_models_v3_2M` (2 M steps, seeds 42-46), `trained_models_v3_g999` (discount factor 0.999), `trained_models_v3_OOL` (variable SERIES operating point) and `trained_models_v4` (remedies; tags `dw`, `rc999`, `rc9999`). A results folder with the suffix `_final` holds the evaluation of the final checkpoints.

## Installation

Python 3.10 or later.

```
pip install -r requirements.txt
```

## Usage

The environment needs the engine and battery maps and, for the full evaluation, the Birmingham cycle. These files are not distributed (see Data). The stored result tables and training logs can be used without them.

Run every command from the repository root.

```
# benchmarks: rule-based controller and A-ECMS on all eight cycles
python baselines_v3.py

# evaluate one trained agent (writes results_check/<run>/summary.csv)
python test_v3.py --run trained_models_v3_2M/GH-SAC-MA_s42_20260927_142937 --out results_check

# train an agent
python train_v3.py --scenario GH-SAC-MA --seed 42 --timesteps 2000000 --save_dir trained_models_v3_2M

# remedy: dwell-aware observation with reward centring at gamma = 0.999
python train_v4.py --scenario GH-SAC-MA --seed 42 --timesteps 2000000 --gamma 0.999 --center 0.001 --tag rc999
python test_v4.py --models trained_models_v4 --out results_v4

# aggregate
python analyze_v3.py
```

The plant and the learned policies are deterministic at evaluation, so `test_v3.py` reproduces the stored `summary.csv` of a run when the maps are in place.

Charge-sustaining-corrected fuel is computed from a summary row as
`total_fuel_g + 245 * 17.5 * (soc0 - final_soc) / 100`.

## Notes on the analysis scripts

The scripts in `analysis_v3/` are the ones used for the paper and are provided as they were run. Call them from the repository root, for example `python analysis_v3/dp_v3.py`. Some of them read the per-step logs or the intermediate checkpoints, which are not part of this repository because of their size (see below).

## Not included

- Per-step logs of the evaluation rollouts (about 0.7 GB) and the intermediate checkpoints saved every 50 000 steps (about 4 GB). `test_v3.py` and `test_v4.py` regenerate the logs; the checkpoints are available on request.
- The battery-model validation of Supplementary Section S1, which uses the public LG 18650HG2 dataset (Kollmeyer et al., Mendeley Data, https://doi.org/10.17632/b5mj79w5w9.2).

## Data

- Included: the speed and wheel-power traces of the standard cycles (NEDC, FTP-75, US06, HWFET, WLTC) and of the two Tehran real-world cycles (`tehran1` = TDC-1, `tehran_shahab` = TDC-2).
- Not distributed: the engine maps (`data_engine/*.data`), the battery open-circuit-voltage and resistance maps (`utiles/OCV.data`, `utiles/Rdch.data`) and the Birmingham cycle (`data_1s/vel_power_birmangham_refine.csv`). They were obtained with a commercial simulation package (Simcenter Amesim) and are not redistributed for licensing reasons. They are available from the corresponding author on reasonable request; `data_engine/README.md` lists where each file belongs.

## Citation

If you use this code, please cite the manuscript above. The full reference will be added on publication.

## License

Code: MIT (see `LICENSE`).
