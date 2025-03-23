@echo off
echo Running Kubernetes Issue Predictor
echo ================================

python -m src.main

echo.
echo Press any key to exit...
pause > nul 