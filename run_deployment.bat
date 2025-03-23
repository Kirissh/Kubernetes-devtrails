@echo off
echo Running Kubernetes Issue Predictor Service
echo ========================================

python -m src.deployment

echo.
echo Press any key to exit...
pause > nul 