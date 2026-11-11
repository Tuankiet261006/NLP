param(
  [switch]$Resume
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Conda = "D:\Tools\Miniconda3\Scripts\conda.exe"
$Environment = "D:\conda-envs\tf-nmt-gpu"
$env:PYTHONUTF8 = "1"
$TrainArguments = @(
  "run", "--no-capture-output", "--prefix", $Environment,
  "python", "-u", "-m", "tensorflow_nmt.tf2_train",
  "--config", "tensorflow_nmt/configs/tf2_gpu.json",
  "--device", "gpu"
)
if ($Resume) {
  $TrainArguments += "--resume"
}

Push-Location $ProjectRoot
try {
  & $Conda @TrainArguments
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
  Pop-Location
}
