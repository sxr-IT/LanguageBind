@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=D:\codex\envs\languagebind\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Python environment not found: %PYTHON%
  pause
  exit /b 1
)
echo Downloading official LanguageBind image and audio weights...
"%PYTHON%" prepare_weights.py
set "RESULT=%ERRORLEVEL%"
echo Exit code: %RESULT%
pause
exit /b %RESULT%
