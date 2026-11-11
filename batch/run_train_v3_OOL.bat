@echo off
REM ================================================================
REM  Reviewer comment 2 -- variable SERIES operating point (optimal operating line, 10-60 kW)
REM  GH-SAC-MA-OOL = G/MA (Gumbel-max + metric-aligned reward) on the OOL plant, 2M steps
REM  Machine A: seeds 42/43/44 in parallel (three windows).
REM  Output: trained_models_v3_OOL\GH-SAC-MA-OOL_s<seed>_<ts>
REM ================================================================
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
start "G/MA-OOL s42" cmd /k python train_v3.py --scenario GH-SAC-MA-OOL --seed 42 --timesteps 2000000 --save_dir trained_models_v3_OOL
start "G/MA-OOL s43" cmd /k python train_v3.py --scenario GH-SAC-MA-OOL --seed 43 --timesteps 2000000 --save_dir trained_models_v3_OOL
start "G/MA-OOL s44" cmd /k python train_v3.py --scenario GH-SAC-MA-OOL --seed 44 --timesteps 2000000 --save_dir trained_models_v3_OOL
