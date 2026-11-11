@echo off
REM ================================================================
REM  Plant v3 -- metric-aligned reward -- machine B: GH-SAC-MA, seeds 42/43/44 in parallel
REM  (opens three windows; each run ~2 h). Output: trained_models_v3\GH-SAC-MA_s<seed>_<ts>
REM ================================================================
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
start "GH-SAC-MA s42" cmd /k python train_v3.py --scenario GH-SAC-MA --seed 42 --timesteps 500000
start "GH-SAC-MA s43" cmd /k python train_v3.py --scenario GH-SAC-MA --seed 43 --timesteps 500000
start "GH-SAC-MA s44" cmd /k python train_v3.py --scenario GH-SAC-MA --seed 44 --timesteps 500000
