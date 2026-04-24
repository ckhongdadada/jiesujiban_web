@echo off
chcp 65001 >nul
echo ============================================================
echo Optuna 参数搜索
echo ============================================================
echo.

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

python tools/optuna_search_classifier.py --n-trials 5 --search-epochs 2 --sample-size 3000 --metric macro_f1

echo.
echo ============================================================
echo 搜索完成
echo ============================================================
echo.
echo 查看结果:
echo   - data\reports\training\optuna\optuna_best_params_latest.json
echo   - data\reports\training\optuna\optuna_trials_latest.csv
echo.
pause
