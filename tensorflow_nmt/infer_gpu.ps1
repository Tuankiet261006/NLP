$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Conda = "D:\Tools\Miniconda3\Scripts\conda.exe"
$Environment = "D:\conda-envs\tf-nmt-gpu"
$env:PYTHONUTF8 = "1"

Push-Location $ProjectRoot
try {
  & $Conda run --no-capture-output --prefix $Environment `
    python -u -m tensorflow_nmt.tf2_infer --device gpu --beam-width 3 `
      --input shared/data/processed/tst2012.en `
      --output tensorflow_nmt/outputs/tf2_gpu/dev.hyp.vi
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

  & $Conda run --no-capture-output --prefix $Environment `
    python -u -m tensorflow_nmt.tf2_infer --device gpu --beam-width 3 `
      --input shared/data/processed/tst2013.en `
      --output tensorflow_nmt/outputs/tf2_gpu/test.hyp.vi
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
  Pop-Location
}
