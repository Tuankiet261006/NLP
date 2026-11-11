$ErrorActionPreference = "Stop"
python "$PSScriptRoot\run_nmt.py" smoke
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

