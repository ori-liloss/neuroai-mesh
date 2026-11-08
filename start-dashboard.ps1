Set-Location -LiteralPath $PSScriptRoot
$py = Join-Path $env:LOCALAPPDATA "Python\pythoncore-3.14-64\python.exe"
if (Test-Path -LiteralPath $py) {
  & $py -u mesh.py serve
} else {
  py -3 -u mesh.py serve
}
