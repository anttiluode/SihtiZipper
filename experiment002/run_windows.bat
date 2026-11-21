@echo off
cd /d "%~dp0.."
python experiment002\compare.py --steps 480 --train-scenes 320 --test-scenes 24
if errorlevel 1 (echo Model run failed; install: python -m pip install numpy torch pillow& pause& exit /b 1)
python experiment002\make_viewer.py
if errorlevel 1 (echo Viewer failed& pause& exit /b 1)
start "" "experiment002\viewer.html"
echo Full receipt: experiment002\results\receipt.json
pause
