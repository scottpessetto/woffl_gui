@echo off
rem ---------------------------------------------------------------------
rem Run the WOFFL web app locally - exactly what a Databricks deploy runs.
rem
rem   Double-click this file, or from a terminal in the repo root:
rem       run_local
rem
rem Builds the SPA (so you are testing the same web/dist a commit would
rem ship), starts the FastAPI server on port 8000, and opens the browser.
rem Ctrl+C in this window stops the server.
rem
rem Optional: `run_local nobrowser` skips opening the browser.
rem ---------------------------------------------------------------------
setlocal
cd /d "%~dp0"

echo [1/2] Building web app...
pushd web
call npm run build
if errorlevel 1 goto :fail
popd

set PY=venv\Scripts\python.exe
if not exist "%PY%" set PY=python

echo [2/2] Starting server at http://127.0.0.1:8000  (Ctrl+C to stop)
if /I not "%~1"=="nobrowser" (
    rem open the browser after the server has had a moment to bind
    start "" cmd /c "timeout /t 2 >nul & start http://127.0.0.1:8000"
)
"%PY%" -m uvicorn server.main:app --port 8000
goto :eof

:fail
popd
echo.
echo Build FAILED - fix the error above before testing.
pause
exit /b 1
