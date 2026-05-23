@echo off
REM ============================================================
REM  SAMet Project Setup  -- uses Python venv (no conda needed)
REM  Prerequisites: Python 3.10+, Git
REM ============================================================

echo [1/5] Creating virtual environment in .venv ...
python -m venv .venv
call .venv\Scripts\activate.bat

echo [2/5] Installing PyTorch with CUDA 11.8...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo [3/5] Installing project requirements...
pip install -r requirements.txt

echo [4/5] Cloning and installing SAM 2...
if not exist sam2_repo (
    git clone https://github.com/facebookresearch/sam2.git sam2_repo
)
cd sam2_repo
pip install -e ".[demo]"
cd ..

echo [5/5] Installing DAVIS evaluation toolkit...
pip install git+https://github.com/davisvideochallenge/davis2017-evaluation.git

echo.
echo Setup complete!
echo To activate the environment next time, run:
echo     .venv\Scripts\activate
echo Then download DAVIS 2017:
echo     python scripts/download_davis.py
