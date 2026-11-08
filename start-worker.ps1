Set-Location -LiteralPath $PSScriptRoot
Write-Output "Leave this window open."
netsh advfirewall firewall add rule name="Neuro worker" dir=in action=allow protocol=TCP localport=8781 profile=private
ipconfig | Select-String "IPv4"
python -u mesh.py work
