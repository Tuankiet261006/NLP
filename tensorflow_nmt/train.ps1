$ErrorActionPreference = "Stop"
python "$PSScriptRoot\run_nmt.py" train
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

