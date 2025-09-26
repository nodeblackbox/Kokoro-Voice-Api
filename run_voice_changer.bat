@echo off
echo Starting Voice Changer API...
echo.

REM Check if conda is installed
if not exist "C:\ProgramData\miniconda3\condabin\conda.bat" (
    echo ERROR: Miniconda not found at C:\ProgramData\miniconda3
    echo Please install Miniconda or update the path
    pause
    exit /b 1
)

REM Activate the specific environment
call C:\ProgramData\miniconda3\condabin\conda.bat activate randnameko3
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to activate environment randnameko3
    pause
    exit /b 1
)

REM Check if PyTorch is installed and CUDA availability
echo Checking PyTorch and CUDA...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: PyTorch not found or import failed. Install with: pip install torch
    pause
    exit /b 1
)

REM Run the voice changer API
echo Starting voicechangerapiV8.py...
echo.
if not exist "voicechangerapiV8.py" (
    echo ERROR: voicechangerapiV8.py not found in current directory
    pause
    exit /b 1
)
python voicechangerapiV8.py
if %ERRORLEVEL% neq 0 (
    echo ERROR: voicechangerapiV8.py failed to run
    pause
    exit /b 1
)

REM Keep the window open if the script exits
echo.
echo Script ended. Press any key to close this window...
pause >nul
