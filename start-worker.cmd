@echo off
cd /d "%~dp0"
netsh advfirewall firewall show rule name="Neuro worker" >nul 2>&1
if errorlevel 1 netsh advfirewall firewall add rule name="Neuro worker" dir=in action=allow protocol=TCP localport=8781 profile=private >nul 2>&1
ipconfig | findstr "IPv4" | findstr /v "169.254." | findstr /v "192.168.56." | findstr /v " 26."
set "PY="
call :pyfile "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
if errorlevel 1 if not errorlevel 2 goto waitpy
if not defined PY call :pyfile "%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
if errorlevel 1 if not errorlevel 2 goto waitpy
if defined PY goto checkpy
where py >nul 2>&1
if %errorlevel%==0 goto checklauncher
set "PY=python"
:checkpy
"%PY%" -c "import socket,sys; s=socket.socket(); s.settimeout(0.4); r=s.connect_ex(('127.0.0.1',8781)); s.close(); sys.exit(0 if r==0 else 1)" >nul 2>&1
if not errorlevel 1 (
  echo The worker is already open on port 8781.
  exit /b 0
)
echo Leave this window open.
"%PY%" -u mesh.py work
if errorlevel 1 pause
exit /b %errorlevel%
:checklauncher
py -3 -c "import socket,sys; s=socket.socket(); s.settimeout(0.4); r=s.connect_ex(('127.0.0.1',8781)); s.close(); sys.exit(0 if r==0 else 1)" >nul 2>&1
if not errorlevel 1 (
  echo The worker is already open on port 8781.
  exit /b 0
)
echo Leave this window open.
py -3 -u mesh.py work
if errorlevel 1 pause
exit /b %errorlevel%
:waitpy
echo Python could not be read. This launch waits.
exit /b 1
:pyfile
set "PYCAND=%~1"
powershell -NoProfile -Command "try { $i = Get-Item -LiteralPath $env:PYCAND -Force -ErrorAction Stop; if ($i.PSIsContainer) { exit 2 }; exit 0 } catch [System.Management.Automation.ItemNotFoundException] { exit 2 } catch { exit 1 }"
if errorlevel 2 exit /b 2
if errorlevel 1 exit /b 1
set "PY=%~1"
exit /b 0
