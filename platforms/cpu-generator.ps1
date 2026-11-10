# Use a training PC's idle CPU and RAM for the data factory: a CPU-only model server plus a worker.
# The GPU stays with training (llama-server runs with -ngl 0; the worker's generate jobs do not
# pause for the GPU lock because NEURO_MODEL_ON_CPU=1).
#   powershell -NoProfile -ExecutionPolicy Bypass -File platforms\cpu-generator.ps1 -Model D:\path\model.gguf
# -Model: any local GGUF (a 4B-9B Q4/Q5 model). -Threads: CPU threads for the model (default: half
# the logical cores). -Slots: parallel answers (default 2). Ctrl+C stops the worker and the model.
param(
  [Parameter(Mandatory = $true)][string]$Model,
  [int]$Threads = 0,
  [int]$Slots = 2,
  [int]$Port = 8090
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo
if (-not (Test-Path -LiteralPath (Join-Path $repo 'mesh.py'))) { throw "mesh.py was not found in $repo." }
if (-not (Test-Path -LiteralPath $Model)) { throw "The model file was not found: $Model" }
if ($Port -eq 8080 -or $Port -lt 1024 -or $Port -gt 65535) { throw 'Use a port other than 8080 (that one is the GPU chat server).' }
if ($Slots -lt 1 -or $Slots -gt 4) { throw '-Slots must be 1 to 4.' }
if ($Threads -le 0) { $Threads = [Math]::Max(2, [int]([Environment]::ProcessorCount / 2)) }

$server = Get-ChildItem -Path (Join-Path $env:LOCALAPPDATA 'Neuro\llama'), 'D:\neuro-models\llama' -Recurse -Filter llama-server.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $server) { throw 'llama-server.exe was not found. Run platforms\windows-worker.cmd once, or put llama.cpp in %LOCALAPPDATA%\Neuro\llama.' }

$py = $null
foreach ($candidate in @((Join-Path $env:LOCALAPPDATA 'Python\pythoncore-3.14-64\python.exe'), (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python314\python.exe'))) {
  if (Test-Path -LiteralPath $candidate) { $py = $candidate; break }
}
if (-not $py) { $py = (Get-Command python -ErrorAction Stop).Source }

$logDir = Join-Path $env:LOCALAPPDATA 'Neuro'
New-Item -ItemType Directory -Force $logDir | Out-Null
$ctx = 8192 * $Slots
$serverArgs = "-m `"$Model`" --host 127.0.0.1 --port $Port -ngl 0 -t $Threads -np $Slots --kv-unified -c $ctx --jinja"
Write-Host "[1/2] Starting the model on the CPU ($Threads threads, $Slots slots) at 127.0.0.1:$Port ..."
$llama = Start-Process $server.FullName -ArgumentList $serverArgs -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput (Join-Path $logDir 'cpu-llama.log') -RedirectStandardError (Join-Path $logDir 'cpu-llama.err.log')
try { $llama.PriorityClass = 'BelowNormal' } catch { }
try {
  $ready = $false
  for ($i = 0; $i -lt 180; $i++) {
    if ($llama.HasExited) { Get-Content (Join-Path $logDir 'cpu-llama.err.log') -Tail 15; throw 'The model stopped.' }
    try { Invoke-WebRequest "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 2 | Out-Null; $ready = $true; break } catch { Start-Sleep -Seconds 2 }
  }
  if (-not $ready) { throw 'The model did not start in 6 minutes.' }
  Write-Host '[2/2] Starting the worker (generate on the CPU; the GPU stays with training).'
  $env:NEURO_LOCAL_LLM = [string]$Port
  $env:NEURO_MODEL_ON_CPU = '1'
  & $py -u mesh.py work
} finally {
  if ($llama -and -not $llama.HasExited) { Stop-Process -Id $llama.Id -Force -ErrorAction SilentlyContinue }
}
