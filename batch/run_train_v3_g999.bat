@echo off
REM ================================================================
REM  Reviewer 2, comment 1 -- discount factor test
REM  G/MA (Gumbel-max + metric-consistent reward) with gamma = 0.999, 2M steps, seeds 42/43/44
REM  Output: trained_models_v3_g999\GH-SAC-MA_g999_s<seed>_<ts>
REM ================================================================
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
start "G/MA g0.999 s42" cmd /k python train_v3.py --scenario GH-SAC-MA --gamma 0.999 --tag g999 --seed 42 --timesteps 2000000 --save_dir trained_models_v3_g999
start "G/MA g0.999 s43" cmd /k python train_v3.py --scenario GH-SAC-MA --gamma 0.999 --tag g999 --seed 43 --timesteps 2000000 --save_dir trained_models_v3_g999
start "G/MA g0.999 s44" cmd /k python train_v3.py --scenario GH-SAC-MA --gamma 0.999 --tag g999 --seed 44 --timesteps 2000000 --save_dir trained_models_v3_g999
