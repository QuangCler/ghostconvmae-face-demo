@echo off
REM Launch the demo (Windows). Assumes requirements installed + checkpoints fetched.
REM -X utf8: the picker labels contain '·', which a cp1252 console cannot encode.
REM -u:      keeps the "Running on local URL" line from sitting in the stdout buffer.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -X utf8 -u app.py %*
) else if exist "venv\Scripts\python.exe" (
  "venv\Scripts\python.exe" -X utf8 -u app.py %*
) else (
  python -X utf8 -u app.py %*
)
