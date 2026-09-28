param(
    [string]$PythonExe = "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python 3.14 was not found: $PythonExe"
}

Write-Host "[SETUP] upgrading pip"
& $PythonExe -m pip install --upgrade pip

Write-Host "[SETUP] installing CUDA 12.8 PyTorch"
& $PythonExe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

Write-Host "[SETUP] installing RT-LCOD dependencies"
& $PythonExe -m pip install -e "$projectRoot[all]"

Write-Host "[SETUP] verifying CUDA"
& $PythonExe -c "import torch; print(f'[CUDA] torch={torch.__version__} available={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

Write-Host "[SETUP] completed. Start Jupyter with: $PythonExe -m jupyter lab"
