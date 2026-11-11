$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Conda = "D:\Tools\Miniconda3\Scripts\conda.exe"
$Environment = "D:\conda-envs\tf-nmt-gpu"
$env:PYTHONUTF8 = "1"

Push-Location $ProjectRoot
try {
  & $Conda run --no-capture-output --prefix $Environment `
    python -u -m tensorflow_nmt.tf2_train --smoke --device gpu
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

  & $Conda run --no-capture-output --prefix $Environment `
    python -u -m tensorflow_nmt.tf2_infer --smoke --device gpu `
      --beam-width 3 --max-sentences 10
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
  Pop-Location
}
