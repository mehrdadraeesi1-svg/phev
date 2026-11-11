@echo off
REM  After run_train_v3_g999.bat has finished: test best and final checkpoints of the gamma=0.999 runs
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OMP_NUM_THREADS=1
python test_v3.py --models trained_models_v3_g999 --out results_v3_g999
python test_v3.py --models trained_models_v3_g999 --out results_v3_g999_final --model_file final_model.zip
pause
