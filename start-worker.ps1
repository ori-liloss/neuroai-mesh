Set-Location -LiteralPath $PSScriptRoot
Write-Output "Leave this window open."
netsh advfirewall firewall add rule name="Neuro worker" dir=in action=allow protocol=TCP localport=8781 profile=private
ipconfig | Select-String "IPv4"
$py = Join-Path $env:LOCALAPPDATA "Python\pythoncore-3.14-64\python.exe"
if (Test-Path -LiteralPath $py) {
  & $py -u mesh.py work
} else {
  py -3 -u mesh.py work
}
