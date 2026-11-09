# Neuro worker for a Windows laptop with an NVIDIA GPU - one command does everything:
#   1. finds Python 3.10+
#   2. downloads llama.cpp for NVIDIA (CUDA 12; Vulkan if nvidia-smi is missing), once
#   3. downloads a small model (Qwen3.5-4B, about 2.9 GB), once, resumable
#   4. starts the model on 127.0.0.1:8080 (this computer only), unless one is already running
#   5. starts the Neuro worker and shows the pairing code
# Run it from CMD:
#   git clone https://github.com/ori-liloss/neuroai-mesh.git %USERPROFILE%\neuroai-mesh
#   %USERPROFILE%\neuroai-mesh\platforms\windows-worker.cmd
# Another main PC address:  set NEURO_MESH_URL=http://192.168.1.20:8780  before the command.
# Files go to %LOCALAPPDATA%\Neuro. Nothing is installed system-wide.
# Ctrl+C stops the worker and the model this script started. If the window is closed instead,
# the model keeps running and the next run reuses it.
# Windows PowerShell 5.1 compatible.

$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
try {
  [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch {
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot
if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot 'mesh.py'))) {
  Write-Host "mesh.py was not found next to this script's folder ($RepoRoot)."
  exit 1
}

$NeuroDir = Join-Path $env:LOCALAPPDATA 'Neuro'
$LlamaDir = Join-Path $NeuroDir 'llama'
$ModelDir = Join-Path $NeuroDir 'models'
$Model = Join-Path $ModelDir 'Qwen3.5-4B-UD-Q4_K_XL.gguf'
$ModelUrl = 'https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-UD-Q4_K_XL.gguf'
$ReleasesUrl = 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=1'
$Log = Join-Path $NeuroDir 'llama-server.log'
$ErrLog = Join-Path $NeuroDir 'llama-server.err.log'
$UserAgent = 'neuro-windows-worker'

# ---------------------------------------------------------------- helpers

function Test-Port([int]$Port) {
  $client = New-Object System.Net.Sockets.TcpClient
  try {
    return $client.ConnectAsync('127.0.0.1', $Port).Wait(400)
  } catch {
    return $false
  } finally {
    $client.Dispose()
  }
}

function Test-ModelUp {
  try {
    $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8080/health' -TimeoutSec 2 -ErrorAction Stop
    return ($r.StatusCode -eq 200)
  } catch {
    return $false
  }
}

function Format-GB([double]$Bytes) {
  return ('{0:N2} GB' -f ($Bytes / 1GB))
}

function Get-CurlExe {
  $c = Get-Command curl.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($c) { return $c.Source }
  return $null
}

# Resumable download to "$Dest.part", renamed to $Dest when complete.
function Get-File([string]$Url, [string]$Dest) {
  $part = "$Dest.part"
  $curl = Get-CurlExe
  if ($curl) {
    & $curl -L -f -C - --retry 3 -o $part $Url
    if ($LASTEXITCODE -ne 0) {
      Write-Host "The download stopped (curl exit $LASTEXITCODE). Run the command again: it resumes."
      return $false
    }
  } else {
    # No curl.exe: HttpWebRequest with a Range header resumes; sizes are printed every 100 MB.
    try {
      $have = 0
      if (Test-Path -LiteralPath $part) { $have = (Get-Item -LiteralPath $part).Length }
      $req = [System.Net.HttpWebRequest]::Create($Url)
      $req.UserAgent = $UserAgent
      $req.AllowAutoRedirect = $true
      if ($have -gt 0) { $req.AddRange([long]$have) }
      $resp = $req.GetResponse()
      $mode = [System.IO.FileMode]::Create
      if ($have -gt 0 -and [int]$resp.StatusCode -eq 206) { $mode = [System.IO.FileMode]::Append } else { $have = 0 }
      $total = $have + $resp.ContentLength
      $in = $resp.GetResponseStream()
      $out = New-Object System.IO.FileStream($part, $mode, [System.IO.FileAccess]::Write)
      try {
        $buf = New-Object byte[] (1MB)
        $done = $have
        $next = $done + 100MB
        while (($n = $in.Read($buf, 0, $buf.Length)) -gt 0) {
          $out.Write($buf, 0, $n)
          $done += $n
          if ($done -ge $next) {
            Write-Host ("      {0} of {1}" -f (Format-GB $done), (Format-GB $total))
            $next = $done + 100MB
          }
        }
      } finally {
        $out.Dispose()
        $in.Close()
        $resp.Close()
      }
    } catch {
      Write-Host "The download stopped: $($_.Exception.Message)"
      Write-Host "Run the command again: it resumes."
      return $false
    }
  }
  if (Test-Path -LiteralPath $Dest) { Remove-Item -LiteralPath $Dest -Force }
  Move-Item -LiteralPath $part -Destination $Dest
  return $true
}

# Only assets[].browser_download_url from the releases LIST (releases/latest has no binaries;
# the release notes contain links, so the JSON text is never searched).
function Get-LlamaAssetUrls([bool]$Nvidia) {
  $releases = Invoke-RestMethod -UseBasicParsing -Uri $ReleasesUrl -Headers @{ 'User-Agent' = $UserAgent; 'Accept' = 'application/vnd.github+json' } -TimeoutSec 30 -ErrorAction Stop
  $urls = @()
  foreach ($rel in @($releases)) {
    foreach ($asset in @($rel.assets)) {
      if ($asset.browser_download_url) { $urls += [string]$asset.browser_download_url }
    }
  }
  if ($Nvidia) {
    # CUDA 12 runs on older drivers than CUDA 13. Prefer 12.4, else another 12.x pair.
    $versions = @('12.4')
    foreach ($u in $urls) {
      if ($u -match '/llama-[^/]+-bin-win-cuda-(12\.\d+)-x64\.zip$' -and $versions -notcontains $Matches[1]) {
        $versions += $Matches[1]
      }
    }
    foreach ($v in $versions) {
      $ve = [regex]::Escape($v)
      $main = $urls | Where-Object { $_ -match "/llama-[^/]+-bin-win-cuda-$ve-x64\.zip$" } | Select-Object -First 1
      $rt = $urls | Where-Object { $_ -match "/cudart-llama-(b\d+-)?bin-win-cuda-$ve-x64\.zip$" } | Select-Object -First 1
      if ($main -and $rt) { return @($main, $rt) }
    }
    return @()
  }
  $vk = $urls | Where-Object { $_ -match '/llama-[^/]+-bin-win-vulkan-x64\.zip$' } | Select-Object -First 1
  if ($vk) { return @($vk) }
  return @()
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

function Find-Server {
  if (-not (Test-Path -LiteralPath $LlamaDir)) { return $null }
  $f = Get-ChildItem -LiteralPath $LlamaDir -Recurse -Filter 'llama-server.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($f) { return $f.FullName }
  return $null
}

function Test-Nvidia {
  if (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue) { return $true }
  $alt = Join-Path $env:ProgramFiles 'NVIDIA Corporation\NVSMI\nvidia-smi.exe'
  return (Test-Path -LiteralPath $alt)
}

function Show-LogTail {
  foreach ($f in @($ErrLog, $Log)) {
    if ((Test-Path -LiteralPath $f) -and (Get-Item -LiteralPath $f).Length -gt 0) {
      Write-Host "      --- last lines of $f ---"
      Get-Content -LiteralPath $f -Tail 15 | ForEach-Object { Write-Host "      $_" }
    }
  }
}

function Test-OutOfMemory {
  $pattern = 'out of memory|cudaMalloc failed|failed to allocate|unable to allocate|OutOfDeviceMemory|not enough memory'
  foreach ($f in @($ErrLog, $Log)) {
    if (Test-Path -LiteralPath $f) {
      if (Select-String -LiteralPath $f -Pattern $pattern -Quiet) { return $true }
    }
  }
  return $false
}

# Returns the started process when /health answers, $null otherwise. Sets $script:LastOom.
function Start-Model([string]$Server, [int]$Ngl) {
  $script:LastOom = $false
  $argLine = "-m `"$Model`" --host 127.0.0.1 --port 8080 -c 8192 --jinja -ngl $Ngl"
  try {
    $proc = Start-Process -FilePath $Server -ArgumentList $argLine -WorkingDirectory (Split-Path -Parent $Server) `
      -WindowStyle Hidden -RedirectStandardOutput $Log -RedirectStandardError $ErrLog -PassThru -ErrorAction Stop
    $null = $proc.Handle  # keeps ExitCode readable after the process ends
  } catch {
    Write-Host "      llama-server could not be started: $($_.Exception.Message)"
    return $null
  }
  $waited = 0
  while (-not (Test-ModelUp)) {
    if ($proc.HasExited) {
      Write-Host "      The model stopped (exit $($proc.ExitCode))."
      Show-LogTail
      $script:LastOom = Test-OutOfMemory
      return $null
    }
    if ($waited -ge 240) {
      Write-Host "      The model did not start in 4 minutes."
      Show-LogTail
      Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
      return $null
    }
    Start-Sleep -Seconds 2
    $waited += 2
  }
  return $proc
}

# ---------------------------------------------------------------- 1. Python

function Resolve-Python {
  $cands = @()
  foreach ($p in @(
    (Join-Path $env:LOCALAPPDATA 'Python\pythoncore-3.14-64\python.exe'),
    (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python314\python.exe')
  )) {
    if (Test-Path -LiteralPath $p -PathType Leaf) { $cands += , @($p) }
  }
  if (Get-Command py -ErrorAction SilentlyContinue) { $cands += , @('py', '-3') }
  if (Get-Command python -ErrorAction SilentlyContinue) { $cands += , @('python') }
  foreach ($c in $cands) {
    $exe = $c[0]
    $rest = @()
    if ($c.Count -gt 1) { $rest = $c[1..($c.Count - 1)] }
    & $exe @rest -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' *> $null
    if ($LASTEXITCODE -eq 0) { return , $c }
  }
  return $null
}

if (Test-Port 8781) {
  Write-Host 'The worker is already open on port 8781.'
  exit 0
}

$Py = Resolve-Python
if (-not $Py) {
  Write-Host 'Python 3.10 or newer was not found. Install it once, then run this again:'
  Write-Host '  winget install Python.Python.3.14'
  Write-Host '(or from https://www.python.org/downloads/windows/ ; open a new CMD window afterwards)'
  exit 1
}
$PyExe = $Py[0]
$PyArgs = @()
if ($Py.Count -gt 1) { $PyArgs = $Py[1..($Py.Count - 1)] }
Write-Host "[1/5] Python: $($Py -join ' ')"

# A "Neuro Worker" icon on the Desktop, so next time is one double-click.
try {
  $desktop = [Environment]::GetFolderPath('Desktop')
  $shortcut = Join-Path $desktop 'Neuro Worker.lnk'
  if ($desktop -and -not (Test-Path -LiteralPath $shortcut)) {
    $link = (New-Object -ComObject WScript.Shell).CreateShortcut($shortcut)
    $link.TargetPath = Join-Path $PSScriptRoot 'windows-worker.cmd'
    $link.WorkingDirectory = $RepoRoot
    $link.Description = 'Start the Neuro worker and its local model'
    $link.Save()
    Write-Host "      Added a 'Neuro Worker' icon to the Desktop. Next time, double-click it."
  }
} catch {
  Write-Host '      (Could not add a Desktop icon; start it with platforms\windows-worker.cmd.)'
}

# ---------------------------------------------------------------- keep awake

$awakeOk = $false
try {
  Add-Type -Namespace Neuro -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] static extern uint SetThreadExecutionState(uint flags);
public static void Awake() { SetThreadExecutionState(0x80000001u); }
public static void Release() { SetThreadExecutionState(0x80000000u); }
'@ -ErrorAction Stop
  $awakeOk = $true
} catch {
  if ('Neuro.Power' -as [type]) { $awakeOk = $true }
}

$script:LlamaProc = $null

# Returns $true when a model answers on 127.0.0.1:8080.
function Invoke-ModelSetup {
  # -------------------------------------------------------------- 2. model
  if (Test-ModelUp) {
    Write-Host '[2/5] A model is already answering on 127.0.0.1:8080. Using it.'
  } else {
    New-Item -ItemType Directory -Force -Path $NeuroDir, $ModelDir | Out-Null
    $server = Find-Server
    if (-not $server) {
      $nvidia = Test-Nvidia
      if ($nvidia) {
        Write-Host '[2/5] Downloading llama.cpp (NVIDIA CUDA 12 build)...'
      } else {
        Write-Host '[2/5] nvidia-smi was not found. Downloading llama.cpp (Vulkan build)...'
      }
      try {
        $assetUrls = @(Get-LlamaAssetUrls $nvidia)
      } catch {
        $assetUrls = @()
        Write-Host "      GitHub did not answer: $($_.Exception.Message)"
      }
      if ($assetUrls.Count -eq 0) {
        Write-Host 'Could not find the llama.cpp download. Check the internet and run again.'
        return $false
      }
      $tmpDir = "$LlamaDir.new"
      $dlDir = Join-Path $NeuroDir 'dl'
      if (Test-Path -LiteralPath $tmpDir) { Remove-Item -LiteralPath $tmpDir -Recurse -Force }
      New-Item -ItemType Directory -Force -Path $tmpDir, $dlDir | Out-Null
      foreach ($u in $assetUrls) {
        Write-Host "      $u"
        $zip = Join-Path $dlDir ([IO.Path]::GetFileName(([Uri]$u).AbsolutePath))
        if (-not (Get-File $u $zip)) { return $false }
        try {
          Expand-Archive -LiteralPath $zip -DestinationPath $tmpDir -Force -ErrorAction Stop
        } catch {
          Write-Host "Could not unpack $zip : $($_.Exception.Message)"
          Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue
          return $false
        }
        Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue
      }
      # The CUDA runtime DLLs must sit next to llama-server.exe.
      $srv = Get-ChildItem -LiteralPath $tmpDir -Recurse -Filter 'llama-server.exe' | Select-Object -First 1
      if ($srv) {
        Get-ChildItem -LiteralPath $tmpDir -Recurse -Filter 'cudart*.dll' | Where-Object { $_.DirectoryName -ne $srv.DirectoryName } |
          ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $srv.DirectoryName -Force }
        Get-ChildItem -LiteralPath $tmpDir -Recurse -Filter 'cublas*.dll' | Where-Object { $_.DirectoryName -ne $srv.DirectoryName } |
          ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $srv.DirectoryName -Force }
      }
      if (Test-Path -LiteralPath $LlamaDir) { Remove-Item -LiteralPath $LlamaDir -Recurse -Force }
      Move-Item -LiteralPath $tmpDir -Destination $LlamaDir
      $server = Find-Server
    } else {
      Write-Host "[2/5] llama.cpp already downloaded: $server"
    }
    if (-not $server) {
      Write-Host 'llama-server.exe was not found after the download.'
      return $false
    }

    if (-not (Test-Path -LiteralPath $Model)) {
      $free = (New-Object System.IO.DriveInfo ([IO.Path]::GetPathRoot($ModelDir))).AvailableFreeSpace
      if ($free -lt 4GB) {
        Write-Host "Only $(Format-GB $free) free on $([IO.Path]::GetPathRoot($ModelDir)). The model needs about 3 GB."
        return $false
      }
      Write-Host '[3/5] Downloading the model (about 2.9 GB). It resumes if interrupted.'
      Write-Host "      $ModelUrl"
      if (-not (Get-File $ModelUrl $Model)) { return $false }
      Write-Host "      Done: $(Format-GB (Get-Item -LiteralPath $Model).Length)"
    } else {
      Write-Host '[3/5] Model already downloaded.'
    }

    Write-Host "      Starting the model on 127.0.0.1:8080 (log: $ErrLog)..."
    $script:LlamaProc = Start-Model $server 99
    if (-not $script:LlamaProc -and $script:LastOom) {
      Write-Host '      Not enough GPU memory for the whole model. Retrying with part of it on the GPU (-ngl 24)...'
      $script:LlamaProc = Start-Model $server 24
    }
    if (-not $script:LlamaProc) {
      Write-Host 'The model could not be started. See the log above.'
      return $false
    }
    Write-Host '      The model is ready.'
  }
  return $true
}

$exitCode = 0
try {
  if ($awakeOk) { [Neuro.Power]::Awake() }
  $modelOk = @(Invoke-ModelSetup)[-1]
  if ($modelOk -ne $true) { exit 1 }

  # -------------------------------------------------------------- 3. firewall
  $principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
  $isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
  netsh advfirewall firewall show rule name="Neuro worker" *> $null
  $ruleExists = ($LASTEXITCODE -eq 0)
  if ($ruleExists) {
    Write-Host '[4/5] Firewall rule "Neuro worker" is present.'
  } elseif ($isAdmin) {
    netsh advfirewall firewall add rule name="Neuro worker" dir=in action=allow protocol=TCP localport=8781 profile=private *> $null
    Write-Host '[4/5] Added firewall rule "Neuro worker" (TCP 8781, private networks only).'
  } else {
    Write-Host '[4/5] The firewall rule is missing. Run this once in a CMD opened as Administrator:'
    Write-Host '      netsh advfirewall firewall add rule name="Neuro worker" dir=in action=allow protocol=TCP localport=8781 profile=private'
    Write-Host '      Continuing without it; the main PC may not reach this laptop until it is added.'
  }

  # -------------------------------------------------------------- 4. worker
  Install-Luau
  if ($env:NEURO_MESH_URL) {
    Write-Host "[5/5] Starting the Neuro worker. Main PC: $env:NEURO_MESH_URL"
  } else {
    Write-Host '[5/5] Starting the Neuro worker. Main PC: the default in mesh.py.'
    Write-Host '      On your network set it first, e.g.  set NEURO_MESH_URL=http://192.168.1.20:8780'
  }
  Write-Host '      A pairing code (Pairing code: ### ###) appears in about 30 seconds.'
  Write-Host "      Approve it on the main PC's Neuro page. Keep the laptop plugged in. Ctrl+C stops everything."
  & $PyExe @PyArgs -u mesh.py work
  $exitCode = $LASTEXITCODE
} finally {
  if ($script:LlamaProc -and -not $script:LlamaProc.HasExited) {
    Write-Host 'Stopping the model this script started...'
    Stop-Process -Id $script:LlamaProc.Id -Force -ErrorAction SilentlyContinue
  }
  if ($awakeOk) { [Neuro.Power]::Release() }
}
exit $exitCode
