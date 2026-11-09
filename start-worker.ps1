Set-Location -LiteralPath $PSScriptRoot
$client = New-Object System.Net.Sockets.TcpClient
try {
  if ($client.ConnectAsync("127.0.0.1", 8781).Wait(400)) {
    Write-Output "The worker is already open on port 8781."
    exit 0
  }
} catch {
} finally {
  $client.Dispose()
}
Write-Output "Leave this window open."
netsh advfirewall firewall show rule name="Neuro worker" *> $null
if ($LASTEXITCODE -ne 0) {
  netsh advfirewall firewall add rule name="Neuro worker" dir=in action=allow protocol=TCP localport=8781 profile=private *> $null
}
ipconfig | Select-String "IPv4" | Where-Object { $_.Line -notmatch "169\.254\.|192\.168\.56\.|: 26\." }
function Resolve-Python {
  foreach ($candidate in @(
    (Join-Path $env:LOCALAPPDATA "Python\pythoncore-3.14-64\python.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\Python\Python314\python.exe")
  )) {
    try {
      $item = Get-Item -LiteralPath $candidate -Force -ErrorAction Stop
    } catch [System.Management.Automation.ItemNotFoundException] {
      continue
    } catch {
      Write-Output "Python could not be read. This launch waits."
      exit 1
    }
    if (-not $item.PSIsContainer) { return $candidate }
  }
  return $null
}
$py = Resolve-Python
if ($py) {
  & $py -u mesh.py work
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
  py -3 -u mesh.py work
} else {
  python -u mesh.py work
}
