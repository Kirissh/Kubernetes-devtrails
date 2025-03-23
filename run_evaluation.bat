@echo off
echo Running Kubernetes Issue Predictor Evaluation
echo ============================================

python -m src.evaluation

echo.
echo Press any key to exit...
pause > nul 