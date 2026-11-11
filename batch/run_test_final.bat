@echo off
REM ================================================================
REM  Reviewer 2, comment 4 -- evaluate the FINAL checkpoint (not the NEDC-selected best model)
REM  of every existing run. Output: results_v3_final\, results_v3_2M_final\, results_v3_OOL_final\
REM  (test only, no training; already-tested runs are skipped)
REM ================================================================
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=1
python test_v3.py --models trained_models_v3     --out results_v3_final     --model_file final_model.zip
python test_v3.py --models trained_models_v3_2M  --out results_v3_2M_final  --model_file final_model.zip
python test_v3.py --models trained_models_v3_OOL --out results_v3_OOL_final --model_file final_model.zip
pause
