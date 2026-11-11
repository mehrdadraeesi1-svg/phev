@echo off
REM Evaluate all v4 runs on the 8 cycles at SOC0 = 50 and 70 %% (best and final checkpoints)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if "%PYTHON%"=="" set PYTHON=python
"%PYTHON%" test_v4.py --models trained_models_v4 --out results_v4
"%PYTHON%" test_v4.py --models trained_models_v4 --out results_v4_final --model_file final_model.zip
echo done
pause
