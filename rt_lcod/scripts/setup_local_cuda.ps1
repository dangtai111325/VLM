param(
    [string]$PythonLauncher = "py"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$virtualEnv = Join-Path $projectRoot ".venv"
$python = Join-Path $virtualEnv "Scripts\python.exe"

if (-not (Test-Path $python)) {
    $launcher = Get-Command $PythonLauncher -ErrorAction SilentlyContinue
    if (-not $launcher) {
        throw "Python Launcher was not found. Install Python 3.11 x64 from python.org, then rerun this script."
    }
    Write-Host "[SETUP] creating virtual environment: $virtualEnv"
    & $launcher.Source -3.11 -m venv $virtualEnv
}

Write-Host "[SETUP] upgrading pip"
& $python -m pip install --upgrade pip

Write-Host "[SETUP] installing CUDA 12.8 PyTorch"
& $python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

Write-Host "[SETUP] installing RT-LCOD dependencies"
& $python -m pip install -e "$projectRoot[all]"

Write-Host "[SETUP] verifying CUDA"
& $python -c "import torch; print(f'[CUDA] torch={torch.__version__} available={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

Write-Host "[SETUP] completed. Start Jupyter with: $virtualEnv\Scripts\jupyter.exe lab"
