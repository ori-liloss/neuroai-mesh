# Installs the NVIDIA CUDA Toolkit (nvcc, runtime, cuBLAS and the other math libraries) for the
# Neuro CUDA checks, without the NVIDIA installer:
#   1. reads NVIDIA's own download list (developer.download.nvidia.com/compute/cuda/redist)
#   2. downloads only the parts a CUDA build needs (about 2.1 GB), checks each file's SHA-256
#   3. unpacks them to %LOCALAPPDATA%\Neuro\cuda\<version> - no admin rights, nothing system-wide
#   4. sets CUDA_PATH and adds its bin folder to PATH for this Windows user
#   5. compiles a small CUDA program with nvcc and runs it on the GPU to prove the whole chain works
# Run it from CMD or PowerShell (the Visual Studio C++ Build Tools must be installed first):
#   %USERPROFILE%\neuroai-mesh\platforms\install-cuda.cmd
# The 12.x line is the default: it still builds for Volta (V100, sm_70) as well as newer GPUs.
#   install-cuda.cmd -Line 13      the newest 13.x instead (no Volta)
#   install-cuda.cmd -Profiler     also Nsight Compute (ncu), for kernel profiling (+325 MB)
#   install-cuda.cmd -SkipRun      build the test but do not run it (the GPU is busy, e.g. training)
#   install-cuda.cmd -Minimal      only nvcc, the runtime and CCCL (enough for the test, ~130 MB)
# Running it again is safe: finished parts are skipped, and the test runs again.
# Windows PowerShell 5.1 compatible.
param(
  [ValidateSet('12', '13')][string]$Line = '12',
  [switch]$Profiler,
  [switch]$SkipRun,
  [switch]$Minimal
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
try {
  [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch {
}

$Base = 'https://developer.download.nvidia.com/compute/cuda/redist/'
$NeuroDir = Join-Path $env:LOCALAPPDATA 'Neuro'
$Parts = @('cuda_nvcc', 'cuda_cudart', 'cuda_cccl', 'cuda_nvrtc', 'cuda_nvtx', 'cuda_profiler_api',
           'libcublas', 'libcurand', 'libcusparse', 'libcusolver', 'libcufft', 'libnvjitlink')
if ($Minimal) { $Parts = @('cuda_nvcc', 'cuda_cudart', 'cuda_cccl') }
if ($Line -eq '13') { $Parts += @('cuda_crt', 'libnvvm') }  # split out of cuda_nvcc in 13.x
if ($Profiler) { $Parts += 'nsight_compute' }

function Say([string]$Text) { Write-Host $Text }
function Fail([string]$Text) { Write-Host ''; Write-Host "FAILED: $Text" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- 1. Visual Studio C++ tools
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vcvars = $null
if (Test-Path -LiteralPath $vswhere) {
  $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  if ($vsPath) {
    $candidate = Join-Path ($vsPath | Select-Object -First 1) 'VC\Auxiliary\Build\vcvars64.bat'
    if (Test-Path -LiteralPath $candidate) { $vcvars = $candidate }
  }
}
if (-not $vcvars) {
  Fail ('The Visual Studio C++ Build Tools were not found; nvcc needs them on Windows. Install them with:' +
        "`n  winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override `"--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended`"" +
        "`nor from https://visualstudio.microsoft.com/visual-cpp-build-tools/ (Desktop development with C++), then run this again.")
}
Say "[1/5] Visual Studio C++ tools: $vcvars"

# ---------------------------------------------------------------- 2. NVIDIA's download list
Say "[2/5] Reading NVIDIA's CUDA download list ($Line.x)..."
try {
  $index = (Invoke-WebRequest -UseBasicParsing -Uri $Base -TimeoutSec 60).Content
} catch {
  Fail "NVIDIA's download server did not answer: $($_.Exception.Message)"
}
$versions = [regex]::Matches($index, "redistrib_($Line\.\d+\.\d+)\.json") | ForEach-Object { $_.Groups[1].Value } |
  Sort-Object -Unique { [version]$_ }
if (-not $versions) { Fail "No CUDA $Line.x release was found at $Base" }
$version = @($versions)[-1]
$manifest = Invoke-WebRequest -UseBasicParsing -Uri ($Base + "redistrib_$version.json") -TimeoutSec 60 | Select-Object -ExpandProperty Content | ConvertFrom-Json
$short = ($version -split '\.')[0..1] -join '.'
$Dest = Join-Path $NeuroDir "cuda\$short"
$Dl = Join-Path $NeuroDir 'cuda-dl'
New-Item -ItemType Directory -Force -Path $Dest, $Dl | Out-Null

$files = @()
foreach ($part in $Parts) {
  $entry = $manifest.$part
  if (-not $entry) { Fail "CUDA $version has no part named $part." }
  $win = $entry.'windows-x86_64'
  if (-not $win) { Fail "CUDA $version has no Windows build of $part." }
  $files += [pscustomobject]@{ Part = $part; Version = $entry.version; Path = $win.relative_path; Sha = $win.sha256; Size = [int64]$win.size }
}
$totalGb = [math]::Round((($files | Measure-Object Size -Sum).Sum) / 1GB, 2)
$drive = Get-PSDrive -Name ($NeuroDir.Substring(0, 1))
$freeGb = [math]::Round($drive.Free / 1GB, 1)
Say "      CUDA $version : $($files.Count) parts, $totalGb GB to download, about $([math]::Round($totalGb * 2.2, 1)) GB unpacked. Free on $($drive.Name): $freeGb GB."
if ($freeGb -lt ($totalGb * 3.5)) { Fail "Not enough free disk space on $($drive.Name): (need about $([math]::Ceiling($totalGb * 3.5)) GB)." }

# ---------------------------------------------------------------- 3. download, check, unpack
$curl = Get-Command curl.exe -ErrorAction SilentlyContinue
$tar = Get-Command tar.exe -ErrorAction SilentlyContinue
$n = 0
foreach ($f in $files) {
  $n += 1
  $marker = Join-Path $Dest ".neuro-part-$($f.Part)-$($f.Version)"
  if (Test-Path -LiteralPath $marker) { Say "[3/5] ($n/$($files.Count)) $($f.Part) $($f.Version): already installed"; continue }
  $zip = Join-Path $Dl ([IO.Path]::GetFileName($f.Path))
  $mb = [math]::Round($f.Size / 1MB)
  Say "[3/5] ($n/$($files.Count)) $($f.Part) $($f.Version) ($mb MB)..."
  $ok = $false
  for ($try = 1; $try -le 3 -and -not $ok; $try++) {
    if (-not ((Test-Path -LiteralPath $zip) -and (Get-Item -LiteralPath $zip).Length -eq $f.Size)) {
      if ($curl) {
        & $curl.Source -L --fail --retry 3 --silent --show-error -C - -o $zip ($Base + $f.Path)
      } else {
        Invoke-WebRequest -UseBasicParsing -Uri ($Base + $f.Path) -OutFile $zip -TimeoutSec 3600
      }
    }
    if ((Test-Path -LiteralPath $zip) -and ((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash -eq $f.Sha.ToUpper())) {
      $ok = $true
    } else {
      Say "      The download did not match NVIDIA's checksum (try $try of 3). Downloading again..."
      Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue
    }
  }
  if (-not $ok) { Fail "$($f.Part) could not be downloaded correctly. Check the internet connection and run this again." }

  $tmp = Join-Path $Dl "unpack-$($f.Part)"
  if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force }
  New-Item -ItemType Directory -Force -Path $tmp | Out-Null
  if ($tar) { & $tar.Source -xf $zip -C $tmp } else { Expand-Archive -LiteralPath $zip -DestinationPath $tmp -Force }
  if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { Fail "$zip could not be unpacked." }
  # Each archive holds one folder (<part>-windows-x86_64-<version>-archive) with bin, include, lib...
  $root = Get-ChildItem -LiteralPath $tmp -Directory | Select-Object -First 1
  if (-not $root) { Fail "$zip was empty." }
  & robocopy.exe $root.FullName $Dest /E /NFL /NDL /NJH /NJS /NP /R:2 /W:2 | Out-Null
  if ($LASTEXITCODE -ge 8) { Fail "Copying $($f.Part) into $Dest failed (robocopy $LASTEXITCODE)." }
  Remove-Item -LiteralPath $tmp -Recurse -Force
  Remove-Item -LiteralPath $zip -Force
  Set-Content -LiteralPath $marker -Value $version -Encoding ascii
}
$nvcc = Join-Path $Dest 'bin\nvcc.exe'
if (-not (Test-Path -LiteralPath $nvcc)) { Fail "nvcc.exe is missing from $Dest\bin after unpacking." }

# ---------------------------------------------------------------- 4. CUDA_PATH and PATH
Say "[4/5] Setting CUDA_PATH and PATH for this Windows user..."
$bin = Join-Path $Dest 'bin'
[Environment]::SetEnvironmentVariable('CUDA_PATH', $Dest, 'User')
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
$kept = @(($userPath -split ';') | Where-Object { $_ -and ($_ -notlike (Join-Path $NeuroDir 'cuda\*')) })
[Environment]::SetEnvironmentVariable('Path', (@($bin) + $kept) -join ';', 'User')
$env:CUDA_PATH = $Dest
$env:Path = "$bin;$env:Path"

# ---------------------------------------------------------------- 5. compile and run a test
Say '[5/5] Compiling a test program with nvcc and running it on the GPU...'
$work = Join-Path $Dl 'test'
New-Item -ItemType Directory -Force -Path $work | Out-Null
$cu = Join-Path $work 'neuro_cuda_test.cu'
$exe = Join-Path $work 'neuro_cuda_test.exe'
@'
#include <cstdio>
#include <cuda_runtime.h>

__global__ void add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  const int n = 1 << 20;
  float *a, *b, *c;
  if (cudaMallocManaged(&a, n * sizeof(float)) != cudaSuccess) { printf("cudaMallocManaged failed\n"); return 1; }
  cudaMallocManaged(&b, n * sizeof(float));
  cudaMallocManaged(&c, n * sizeof(float));
  for (int i = 0; i < n; ++i) { a[i] = 1.0f * i; b[i] = 2.0f; }
  add<<<(n + 255) / 256, 256>>>(a, b, c, n);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { printf("kernel failed: %s\n", cudaGetErrorString(err)); return 1; }
  for (int i = 0; i < n; ++i) {
    if (c[i] != a[i] + 2.0f) { printf("wrong result at %d\n", i); return 1; }
  }
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  printf("CUDA OK: %s (sm_%d%d), %d numbers added on the GPU\n", p.name, p.major, p.minor, n);
  return 0;
}
'@ | Set-Content -LiteralPath $cu -Encoding ascii

function Build([string]$Extra) {
  $bat = Join-Path $work 'build.bat'
  @(
    '@echo off',
    "call `"$vcvars`" >nul 2>&1",
    "`"$nvcc`" -O2 $Extra -o `"$exe`" `"$cu`" 2>&1"
  ) | Set-Content -LiteralPath $bat -Encoding ascii
  # The 2>&1 is inside the .bat: in Windows PowerShell 5.1 a native program's stderr redirected here
  # becomes an error record, and with $ErrorActionPreference = 'Stop' a compiler warning would end the script.
  $out = & cmd.exe /d /c "`"$bat`"" | Out-String
  return @{ Code = $LASTEXITCODE; Out = $out }
}
$flags = ''
$r = Build ''
if ($r.Code -ne 0 -and $r.Out -match 'unsupported Microsoft Visual Studio version') {
  Say '      This Visual Studio is newer than CUDA officially supports. Building again with -allow-unsupported-compiler...'
  $flags = '-allow-unsupported-compiler'
  $r = Build $flags
}
if ($r.Code -ne 0) { Write-Host $r.Out; Fail 'nvcc could not compile the test program (output above).' }
Set-Content -LiteralPath (Join-Path $Dest 'neuro-nvcc-flags.txt') -Value $flags -Encoding ascii
if ($SkipRun) {
  $run = 'CUDA build OK (the test program was not run: -SkipRun).'
} else {
  $run = & $exe | Out-String
  if ($LASTEXITCODE -ne 0 -or $run -notmatch 'CUDA OK') {
    Write-Host $run
    Fail 'The test program was built but did not run correctly on the GPU (output above). Is the NVIDIA driver up to date?'
  }
}
Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue

$nvccVersion = (& $nvcc --version | Select-String 'release').Line
Write-Host ''
Write-Host "Done. $($run.Trim())" -ForegroundColor Green
Write-Host "  nvcc: $nvcc"
Write-Host "  $nvccVersion"
Write-Host '  CUDA_PATH and PATH are set for this user; new windows find nvcc by name.'
Write-Host '  Start the worker again so it picks up nvcc: platforms\windows-worker.cmd'
exit 0
