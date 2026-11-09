Set-Location -LiteralPath $PSScriptRoot
$client = New-Object System.Net.Sockets.TcpClient
try {
  if ($client.ConnectAsync("127.0.0.1", 8780).Wait(400)) {
    Write-Output "The page is already open on port 8780."
    exit 0
  }
} catch {
} finally {
  $client.Dispose()
}
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
  & $py -u mesh.py serve
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
  py -3 -u mesh.py serve
} else {
  python -u mesh.py serve
}
