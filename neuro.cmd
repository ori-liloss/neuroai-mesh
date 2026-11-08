@echo off
setlocal
cd /d D:\liliai\neuro-cluster
if /I "%~1"=="tools" (
  echo LIST GREP READ WRITE MKDIR RUN
  exit /b 0
)
set "PY=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" -u mesh.py code %*
