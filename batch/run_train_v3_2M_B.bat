@echo off
REM ================================================================
REM  Plant v3 -- LONG training (2M steps) -- machine B: GH-SAC-MA, seeds 42/43/44 in parallel
REM  (three windows; ~4x the 500k runtime). Output: trained_models_v3_2M\GH-SAC-MA_s<seed>_<ts>
REM  Afterwards:  python test_v3.py --models trained_models_v3_2M --out results_v3_2M
REM ================================================================
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
start "GH-SAC-MA s42 2M" cmd /k python train_v3.py --scenario GH-SAC-MA --seed 42 --timesteps 2000000 --save_dir trained_models_v3_2M
start "GH-SAC-MA s43 2M" cmd /k python train_v3.py --scenario GH-SAC-MA --seed 43 --timesteps 2000000 --save_dir trained_models_v3_2M
start "GH-SAC-MA s44 2M" cmd /k python train_v3.py --scenario GH-SAC-MA --seed 44 --timesteps 2000000 --save_dir trained_models_v3_2M
