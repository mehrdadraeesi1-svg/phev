@echo off
REM ================================================================
REM  Reviewer comment 3 -- two extra seeds (45, 46) for the 2M-step metric-aligned agents
REM  -> five seeds each for G/MA and A/MA.  Machine B: four windows in parallel.
REM  Output: trained_models_v3_2M\<scenario>_s<seed>_<ts>  (same folder as seeds 42-44)
REM ================================================================
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
start "G/MA s45 2M" cmd /k python train_v3.py --scenario GH-SAC-MA --seed 45 --timesteps 2000000 --save_dir trained_models_v3_2M
start "G/MA s46 2M" cmd /k python train_v3.py --scenario GH-SAC-MA --seed 46 --timesteps 2000000 --save_dir trained_models_v3_2M
start "A/MA s45 2M" cmd /k python train_v3.py --scenario DH-SAC-MA --seed 45 --timesteps 2000000 --save_dir trained_models_v3_2M
start "A/MA s46 2M" cmd /k python train_v3.py --scenario DH-SAC-MA --seed 46 --timesteps 2000000 --save_dir trained_models_v3_2M
