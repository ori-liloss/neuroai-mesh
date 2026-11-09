@echo off
rem Neuro worker for a Windows laptop with an NVIDIA GPU. See windows-worker.ps1.
rem   git clone https://github.com/ori-liloss/neuroai-mesh.git %USERPROFILE%\neuroai-mesh
rem   %USERPROFILE%\neuroai-mesh\platforms\windows-worker.cmd
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0windows-worker.ps1" %*
set "RC=%errorlevel%"
if not "%RC%"=="0" pause
exit /b %RC%
