@echo off
cd /d "%~dp0"
echo Leave this window open.
netsh advfirewall firewall add rule name="Neuro worker" dir=in action=allow protocol=TCP localport=8781 profile=private
ipconfig | findstr IPv4
python -u mesh.py work
if errorlevel 1 py -3 -u mesh.py work
