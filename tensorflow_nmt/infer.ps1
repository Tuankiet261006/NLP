$ErrorActionPreference = "Stop"
python "$PSScriptRoot\run_nmt.py" infer
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

