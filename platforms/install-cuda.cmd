@echo off
rem Installs the NVIDIA CUDA Toolkit for the Neuro CUDA checks. See install-cuda.ps1.
rem   %USERPROFILE%\neuroai-mesh\platforms\install-cuda.cmd
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-cuda.ps1" %*
set "RC=%errorlevel%"
pause
exit /b %RC%
