@echo off
cd /d "%~dp0"
echo Leave this window open.
netsh advfirewall firewall add rule name="Neuro worker" dir=in action=allow protocol=TCP localport=8781 profile=private
ipconfig | findstr IPv4
where py >nul 2>&1
if %errorlevel%==0 goto usepy
python -u mesh.py work
exit /b %errorlevel%
:usepy
py -3 -u mesh.py work
exit /b %errorlevel%
