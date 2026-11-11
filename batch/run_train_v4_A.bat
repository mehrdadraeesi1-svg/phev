@echo off
REM v4 proposed method, lane A (dw): GH-SAC-MA + dwell-aware observation + refused-request penalty
REM gamma=0.99, reward centering eta=0 (0 = off). 3 seeds x 2M steps.
REM If your packages live in a conda/venv environment, activate it first, or set PYTHON below
REM to the full path of that python.exe, e.g.  set PYTHON=C:\Users\user\anaconda3\envs\rl\python.exe
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if "%PYTHON%"=="" set PYTHON=python
"%PYTHON%" -c "import numpy, pandas, scipy, gymnasium, torch, stable_baselines3" 2>nul
if errorlevel 1 (
  echo.
  echo [ERROR] "%PYTHON%" does not have the required packages.
  echo         Activate the environment used for the v3 runs, or install them with:
  echo         "%PYTHON%" -m pip install -r requirements_v4.txt
  pause
  exit /b 1
)
for %%s in (42 43 44) do (
  echo === lane A tag dw seed %%s
  "%PYTHON%" train_v4.py --scenario GH-SAC-MA --seed %%s --timesteps 2000000 --gamma 0.99 --center 0 --tag dw --save_dir trained_models_v4 --threads 1
)
echo lane A finished
pause
