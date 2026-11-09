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
# verify_luau jobs need the luau command line tool. If it is missing, fetch the official
# release once into %LOCALAPPDATA%\Neuro\luau (where mesh.py looks after NEURO_LUAU and PATH).
# Opt out with  set NEURO_NO_LUAU_DOWNLOAD=1 . Any failure only means no verify_luau jobs here.
function Get-LuauAssetUrl {
  $releases = Invoke-RestMethod -UseBasicParsing -Uri 'https://api.github.com/repos/luau-lang/luau/releases?per_page=1' `
    -Headers @{ 'User-Agent' = 'neuro-worker'; 'Accept' = 'application/vnd.github+json' } -TimeoutSec 30 -ErrorAction Stop
  # Only assets[].browser_download_url: the release notes are never searched.
  foreach ($rel in @($releases)) {
    foreach ($asset in @($rel.assets)) {
      $u = [string]$asset.browser_download_url
      if ($u -match '^https://github\.com/luau-lang/luau/releases/download/[^/]+/luau-windows\.zip$') { return $u }
    }
  }
  return $null
}

function Install-Luau {
  if ($env:NEURO_NO_LUAU_DOWNLOAD -and $env:NEURO_NO_LUAU_DOWNLOAD -ne '0') { return }
  $dir = Join-Path $env:LOCALAPPDATA 'Neuro\luau'
  $exe = Join-Path $dir 'luau.exe'
  $here = (Get-Location).Path
  foreach ($have in @($env:NEURO_LUAU, $env:NEURO_LUAU_BIN, $exe, (Join-Path $here 'tools\luau.exe'), (Join-Path $here 'luau.exe'))) {
    if ($have -and (Test-Path -LiteralPath $have -PathType Leaf)) { return }
  }
  if (Get-Command luau -ErrorAction SilentlyContinue) { return }
  $ProgressPreference = 'SilentlyContinue'
  try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
  } catch {
  }
  $zip = Join-Path $dir 'luau-windows.zip.part'
  try {
    Write-Host '      luau was not found. Downloading the official release for verify_luau jobs (set NEURO_NO_LUAU_DOWNLOAD=1 to skip)...'
    $url = Get-LuauAssetUrl
    if (-not $url) { throw 'no luau-windows.zip in the latest release' }
    Write-Host "      $url"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $zip -Headers @{ 'User-Agent' = 'neuro-worker' } -TimeoutSec 120 -ErrorAction Stop
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($zip)
    try {
      # Only the one file, by name: no other path from the archive is written.
      $entry = $archive.Entries | Where-Object { $_.Name -eq 'luau.exe' } | Select-Object -First 1
      if (-not $entry) { throw 'luau.exe is not in the download' }
      [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, "$exe.part", $true)
    } finally {
      $archive.Dispose()
    }
    Move-Item -LiteralPath "$exe.part" -Destination $exe -Force
    Write-Host "      luau is in $exe"
  } catch {
    Write-Host "      luau download skipped: $($_.Exception.Message). The worker runs without verify_luau jobs."
    Remove-Item -LiteralPath "$exe.part" -Force -ErrorAction SilentlyContinue
  } finally {
    Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue
  }
}
Install-Luau
$py = Resolve-Python
if ($py) {
  & $py -u mesh.py work
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
  py -3 -u mesh.py work
} else {
  python -u mesh.py work
}
