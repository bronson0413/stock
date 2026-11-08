@echo off
REM One-click real-data run for Windows. Open "Anaconda Prompt" (or any prompt
REM where `python` works), cd into the repo folder, then:  scripts\run_tw.bat
REM
REM Stage 1: random 500 stocks (incl. delisted)  -> reports\tw500   (~2-3 h with a token)
REM Stage 2: full market, reusing stage 1's cache -> reports\tw_full (overnight)
REM Safe to close and re-run at any time: every download is cached in data\cache.

setlocal
cd /d "%~dp0\.."

if "%FINMIND_TOKEN%"=="" (
  echo A free FinMind token doubles the hourly request quota ^(300 -^> 600^).
  echo Register at https://finmindtrade.com and paste the token, or press Enter to skip.
  set /p FINMIND_TOKEN=FinMind token:
)

python -m pip install -q -e ".[dev,ml]" || goto :fail
python -m pytest -q || goto :fail

set ARGS=--start 2013-01-01 --end 2026-10-31 --holdout-start 2023-01-01

echo.
echo === Stage 1: random sample of 500 stocks ===
python -m twfactor finmind %ARGS% --sample 500 --out reports\tw500 || goto :fail
python -m twfactor finmind %ARGS% --sample 500 --industry-neutral --out reports\tw500_ind || goto :fail

echo.
echo === Stage 2: full market ===
python -m twfactor finmind %ARGS% --out reports\tw_full || goto :fail
python -m twfactor finmind %ARGS% --industry-neutral --out reports\tw_full_ind || goto :fail

echo.
echo Done. Reports are in the reports folder.
exit /b 0

:fail
echo.
echo Stopped with an error (see above). Re-running continues from the cache.
exit /b 1
