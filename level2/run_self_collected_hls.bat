@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not defined VIVADO_HLS_ROOT set "VIVADO_HLS_ROOT=E:\use\cpu\Vivado\2019.2"
if not exist "results\self_collected\preprocessed_self_collected.bin" (
  echo ERROR: Run evaluate-dataset first.
  exit /b 2
)
call "%VIVADO_HLS_ROOT%\settings64.bat"
set "LENET_DATA_W=16"
set "LENET_SKIP_SYNTH=1"
set "LENET_SKIP_CSIM=0"
set "LENET_ACCURACY_THRESHOLD=0"
set "LENET_HLS_WORKSPACE=%CD%\hls_work\self_collected_w16"
set "LENET_ACCURACY_BLOB=%CD%\results\self_collected\preprocessed_self_collected.bin"
set "LENET_RESULT_CSV=%CD%\results\self_collected\hls_results.csv"
call "%VIVADO_HLS_ROOT%\bin\vivado_hls.bat" -f ..\level1\run_hls.tcl
if errorlevel 1 exit /b %errorlevel%
python ..\level1\tools\lenet_validation.py compare ^
  --float-results results\self_collected\python_results.csv ^
  --hls-results results\self_collected\hls_results.csv ^
  --report results\self_collected\python_hls_comparison.json ^
  --mismatches results\self_collected\python_hls_mismatches.csv ^
  --threshold 0
if errorlevel 1 exit /b %errorlevel%
python tools\level2_validation.py finalize-hls ^
  --summary results\self_collected\summary.json ^
  --comparison results\self_collected\python_hls_comparison.json ^
  --report results\self_collected\experiment_report.md ^
  --tool-version "Vitis HLS 2025.2.1"
exit /b %errorlevel%
