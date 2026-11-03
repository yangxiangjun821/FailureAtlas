$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Create the Python 3.12 virtual environment before running this script."
}

& $Python -m pip install --timeout 300 --retries 5 -r (Join-Path $ProjectRoot "requirements-base.txt")
& $Python -m pip install --timeout 1000 --retries 10 -r (Join-Path $ProjectRoot "requirements-gpu.txt")
& $Python (Join-Path $ProjectRoot "verify_environment.py")

