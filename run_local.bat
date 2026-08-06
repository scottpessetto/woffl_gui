@echo off
rem ---------------------------------------------------------------------
rem Run the WOFFL web app locally - exactly what a Databricks deploy runs.
rem
rem   Double-click this file, or from a terminal in the repo root:
rem       run_local
rem
rem Builds the SPA (so you are testing the same web/dist a commit would
rem ship), stops any previous instance still holding port 8000, starts the
rem FastAPI server on that port, and opens the browser.
rem Ctrl+C in this window stops the server.
rem
rem Optional: `run_local nobrowser` skips opening the browser.
rem ---------------------------------------------------------------------
setlocal
cd /d "%~dp0"

echo [1/3] Building web app...
pushd web
call npm run build
if errorlevel 1 goto :fail
popd

set PY=venv\Scripts\python.exe
if not exist "%PY%" set PY=python

rem Kill anything still LISTENING on port 8000 - forgotten instances from
rem earlier runs. Scoped to the listener PID, never other python processes.
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:":8000 .*LISTENING"') do (
    echo [2/3] Stopping previous instance (PID %%p^)...
    taskkill /f /t /pid %%p >nul 2>&1
)

echo [3/3] Starting server at http://127.0.0.1:8000  (Ctrl+C to stop)
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
