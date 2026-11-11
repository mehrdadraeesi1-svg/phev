@echo off
REM ================================================================
REM  PHEV plant v3 training -- lane A3 (machine A) -- scenario GH-SAC
REM  Unified environment PHEVEnv_v13 (fixed 3.40 coupled ratio, 30-min engine
REM  cool-down, 10-s minimum dwell, GHG-only WTW, 1-s cycles in data_1s).
REM  Output: trained_models_v3\GH-SAC_s<seed>_<timestamp>
REM ================================================================
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1

echo [1/3] GH-SAC seed=42
python train_v3.py --scenario GH-SAC --seed 42 --timesteps 500000
echo [2/3] GH-SAC seed=43
python train_v3.py --scenario GH-SAC --seed 43 --timesteps 500000
echo [3/3] GH-SAC seed=44
python train_v3.py --scenario GH-SAC --seed 44 --timesteps 500000

echo LANE A3 DONE
pause
