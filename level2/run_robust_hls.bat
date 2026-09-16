@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not defined VIVADO_HLS_ROOT (
  echo Set VIVADO_HLS_ROOT to your actual Vivado HLS installation first.
  exit /b 2
)
if not exist "%VIVADO_HLS_ROOT%\bin\vivado_hls.bat" exit /b 2
if not exist "results\preprocessing_v1\preprocessed_accepted.bin" exit /b 2
if exist "results\preprocessing_v1\hls_results.csv" (
  echo Existing HLS result found. Archive it before a deliberate rerun.
  exit /b 2
)
call "%VIVADO_HLS_ROOT%\settings64.bat"
if errorlevel 1 exit /b %errorlevel%
set "LENET_DATA_W=16"
set "LENET_SKIP_SYNTH=1"
set "LENET_SKIP_CSIM=0"
set "LENET_ACCURACY_THRESHOLD=0"
set "LENET_HLS_WORKSPACE=%CD%\hls_work\robust_v1_w16"
set "LENET_ACCURACY_BLOB=%CD%\results\preprocessing_v1\preprocessed_accepted.bin"
set "LENET_RESULT_CSV=%CD%\results\preprocessing_v1\hls_results.csv"
call "%VIVADO_HLS_ROOT%\bin\vivado_hls.bat" -f ..\level1\run_hls.tcl
if errorlevel 1 exit /b %errorlevel%
python ..\level1\tools\lenet_validation.py compare --float-results results\preprocessing_v1\python_results.csv --hls-results results\preprocessing_v1\hls_results.csv --report results\preprocessing_v1\python_hls_comparison.json --mismatches results\preprocessing_v1\python_hls_mismatches.csv --threshold 0
echo Compare reports cover accepted inputs only. Include the 8 rejections as errors in end-to-end accuracy over 503.
exit /b %errorlevel%
