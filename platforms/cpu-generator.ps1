# Use a training PC's idle CPU and RAM for the data factory: a CPU-only model server plus a worker,
# while the GPU stays with training.
#   powershell -NoProfile -ExecutionPolicy Bypass -File platforms\cpu-generator.ps1 -Model D:\path\model.gguf
# -Model: a local GGUF (a 4B Q4 model is the right size). -Threads: CPU threads for the model (default 8).
# -Slots: parallel answers (default 2). Ctrl+C stops the worker and the model.
#
# Safe next to a training run on the same PC:
# - Only a llama.cpp build WITHOUT ggml-cuda.dll / ggml-vulkan.dll is used. A CUDA build run with
#   -ngl 0 still creates a CUDA context and sends big matmuls to the GPU; that VRAM once pushed a
#   training stage into CUDA out-of-memory. CUDA_VISIBLE_DEVICES=-1 and --device none are set as well,
#   and nvidia-smi must not list the model process, or it is stopped at once.
# - Memory guard: the model stops when free commit memory drops under -MinFreeGB (default 6) or free
#   RAM under 2 GB (a stage's data build needs ~10 GB), and starts again above -ResumeGB (default 10).
#   The worker keeps running and does check jobs meanwhile.
# - The model and the worker run at below-normal priority.
# Log: %LOCALAPPDATA%\Neuro\cpu-generator.log (worker output: cpu-worker.log).
param(
  [Parameter(Mandatory = $true)][string]$Model,
  [int]$Threads = 8,
  [int]$Slots = 2,
  [int]$Port = 8090,
  [double]$MinFreeGB = 6,
  [double]$ResumeGB = 10
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo
if (-not (Test-Path -LiteralPath (Join-Path $repo 'mesh.py'))) { throw "mesh.py was not found in $repo." }
if (-not (Test-Path -LiteralPath $Model)) { throw "The model file was not found: $Model" }
if ($Port -eq 8080 -or $Port -lt 1024 -or $Port -gt 65535) { throw 'Use a port other than 8080 (that one is the GPU chat server).' }
if ($Slots -lt 1 -or $Slots -gt 4) { throw '-Slots must be 1 to 4.' }
if ($Threads -le 0) { $Threads = [Math]::Max(2, [int]([Environment]::ProcessorCount / 3)) }

$logDir = Join-Path $env:LOCALAPPDATA 'Neuro'
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir 'cpu-generator.log'
function Say([string]$Text) {
  $line = "$(Get-Date -Format s) $Text"
  Write-Host $line
  Add-Content -LiteralPath $log -Value $line -Encoding UTF8
}

# ---------------------------------------------------------------- a CPU-only llama.cpp build
$server = $null
# Newest first: an old build (b7508 in llama-cpp-win) does not know Qwen3.5 ("unknown model architecture: 'qwen35'").
foreach ($dir in @('D:\liliai\tools\llama-cpp-cpu', 'D:\liliai\tools\llama-cpp-win', (Join-Path $env:LOCALAPPDATA 'Neuro\llama'), 'D:\neuro-models\llama')) {
  foreach ($exe in @(Get-ChildItem -Path $dir -Recurse -Filter llama-server.exe -ErrorAction SilentlyContinue)) {
    $folder = $exe.DirectoryName
    $gpuDlls = @(Get-ChildItem -LiteralPath $folder -Filter 'ggml-*.dll' -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -match 'cuda|vulkan|sycl|hip|opencl|kompute' })
    if ($gpuDlls.Count -eq 0) { $server = $exe.FullName; break }
  }
  if ($server) { break }
}
if (-not $server) { throw 'No CPU-only llama.cpp build was found (one without ggml-cuda.dll / ggml-vulkan.dll). A GPU build is not used next to training.' }

$py = $null
foreach ($candidate in @((Join-Path $env:LOCALAPPDATA 'Python\pythoncore-3.14-64\python.exe'), (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python314\python.exe'))) {
  if (Test-Path -LiteralPath $candidate) { $py = $candidate; break }
}
if (-not $py) { $py = (Get-Command python -ErrorAction Stop).Source }

# The model process inherits these: no GPU is visible to it at all.
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:GGML_VK_VISIBLE_DEVICES = ''
$help = (cmd.exe /d /c "`"$server`" --help 2>&1") -join "`n"
$ctx = 8192 * $Slots
$serverArgs = "-m `"$Model`" --host 127.0.0.1 --port $Port -ngl 0 -t $Threads -np $Slots --kv-unified -c $ctx --jinja"
if ($help -match '(^|[\s,])--device([\s,=]|$)') { $serverArgs += ' --device none' }
if ($help -match '(^|[\s,])--cache-ram([\s,=]|$)') { $serverArgs += ' --cache-ram 0' }

function Get-Memory {
  $c = (Get-Counter '\Memory\Committed Bytes', '\Memory\Commit Limit', '\Memory\Available MBytes').CounterSamples
  return @{ FreeCommitGB = ($c[1].CookedValue - $c[0].CookedValue) / 1GB; AvailGB = $c[2].CookedValue / 1KB }
}

function Test-OnGpu([int]$ProcessId) {
  $smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
  if (-not $smi) { return $false }
  $pids = & $smi.Source --query-compute-apps=pid --format=csv,noheader 2>$null
  return [bool](@($pids) | Where-Object { "$_".Trim() -eq "$ProcessId" })
}

$script:Llama = $null
function Start-CpuModel {
  $script:Llama = Start-Process $server -ArgumentList $serverArgs -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logDir 'cpu-llama.log') -RedirectStandardError (Join-Path $logDir 'cpu-llama.err.log')
  try { $script:Llama.PriorityClass = 'BelowNormal' } catch { }
  for ($i = 0; $i -lt 180; $i++) {
    if ($script:Llama.HasExited) {
      Say "The CPU model stopped while loading (exit $($script:Llama.ExitCode)). Last lines:"
      Get-Content (Join-Path $logDir 'cpu-llama.err.log') -Tail 10 | ForEach-Object { Say "  $_" }
      if (Select-String -LiteralPath (Join-Path $logDir 'cpu-llama.err.log') -Pattern 'unknown model architecture' -Quiet) {
        Say "This llama.cpp build ($server) is too old for the model. Put a newer CPU build (llama-bNNNN-bin-win-cpu-x64.zip) in D:\liliai\tools\llama-cpp-cpu."
      }
      $script:Llama = $null
      return $false
    }
    try {
      Invoke-WebRequest "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 2 | Out-Null
      if (Test-OnGpu $script:Llama.Id) {
        Say 'SAFETY: the CPU model appeared on the GPU (nvidia-smi lists it). Stopping it; nothing more is started.'
        Stop-CpuModel
        throw 'The CPU model used the GPU.'
      }
      Say "CPU model ready: $Threads threads, $Slots slots, port $Port, pid $($script:Llama.Id)."
      return $true
    } catch [System.Net.WebException] {
      Start-Sleep -Seconds 2
    }
  }
  Say 'The CPU model did not start in 6 minutes.'
  Stop-CpuModel
  return $false
}
function Stop-CpuModel {
  if ($script:Llama -and -not $script:Llama.HasExited) { Stop-Process -Id $script:Llama.Id -Force -ErrorAction SilentlyContinue }
  $script:Llama = $null
}

Say "Starting. Server: $server. Model: $Model. Guard: stop under $MinFreeGB GB free commit or 2 GB free RAM, resume above $ResumeGB GB."
$worker = $null
try {
  $mem = Get-Memory
  if ($mem.FreeCommitGB -gt $ResumeGB -and $mem.AvailGB -gt 4) { [void](Start-CpuModel) }
  else { Say ("Not enough free memory to start the model now ({0:N1} GB commit, {1:N1} GB RAM free); waiting." -f $mem.FreeCommitGB, $mem.AvailGB) }

  $env:NEURO_LOCAL_LLM = [string]$Port
  $env:NEURO_MODEL_ON_CPU = '1'
  $worker = Start-Process -FilePath $py -ArgumentList '-u', 'mesh.py', 'work' -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logDir 'cpu-worker.log') -RedirectStandardError (Join-Path $logDir 'cpu-worker.err.log')
  try { $worker.PriorityClass = 'BelowNormal' } catch { }
  Say "Worker started (pid $($worker.Id)): generate on the CPU model, plus check jobs."

  $failures = 0
  while (-not $worker.HasExited) {
    Start-Sleep -Seconds 15
    $mem = Get-Memory
    $running = $script:Llama -and -not $script:Llama.HasExited
    if ($running -and ($mem.FreeCommitGB -lt $MinFreeGB -or $mem.AvailGB -lt 2)) {
      Say ("Memory guard: {0:N1} GB commit / {1:N1} GB RAM free. Stopping the CPU model." -f $mem.FreeCommitGB, $mem.AvailGB)
      Stop-CpuModel
    } elseif (-not $running -and $mem.FreeCommitGB -gt $ResumeGB -and $mem.AvailGB -gt 4) {
      if ($script:Llama) { Say "The CPU model exited (exit $($script:Llama.ExitCode)). Starting it again." }
      if (Start-CpuModel) { $failures = 0 }
      else {
        $failures += 1
        if ($failures -ge 3) { Say 'The CPU model failed to start 3 times in a row. Stopping.'; break }
        Start-Sleep -Seconds 60
      }
    } elseif ($running -and (Test-OnGpu $script:Llama.Id)) {
      Say 'SAFETY: the CPU model appeared on the GPU. Stopping everything.'
      break
    }
  }
  if ($worker.HasExited) { Say "The worker exited (code $($worker.ExitCode)). See cpu-worker.log." }
} finally {
  Stop-CpuModel
  if ($worker -and -not $worker.HasExited) { Stop-Process -Id $worker.Id -Force -ErrorAction SilentlyContinue }
  Say 'Stopped.'
}
