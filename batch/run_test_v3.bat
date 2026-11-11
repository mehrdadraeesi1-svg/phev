@echo off
REM  Test every trained plant-v3 run (skips runs already tested) -> results_v3\<run>\summary.csv
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=1
python test_v3.py
python analyze_v3.py
pause
