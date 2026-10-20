@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=D:\codex\envs\languagebind\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Python environment not found: %PYTHON%
  pause
  exit /b 1
)
echo Running LanguageBind CPU demo...
"%PYTHON%" run_cpu_demo.py
set "RESULT=%ERRORLEVEL%"
echo Exit code: %RESULT%
pause
exit /b %RESULT%
