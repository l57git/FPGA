@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not defined VIVADO_HLS_ROOT set "VIVADO_HLS_ROOT=E:\use\cpu\Vivado\2019.2"
if not exist "results\stress_test\preprocessed_stress_test.bin" (
  echo ERROR: Run the Level 2 benchmark first.
  exit /b 2
)
call "%VIVADO_HLS_ROOT%\settings64.bat"
set "PATH=%VIVADO_HLS_ROOT%\tps\win64\msys64\mingw64\bin;%VIVADO_HLS_ROOT%\tps\mingw\6.2.0\win64.o\nt\bin;%PATH%"
set "LENET_SKIP_SYNTH=1"
set "LENET_ACCURACY_BLOB=%CD%\results\stress_test\preprocessed_stress_test.bin"
set "LENET_RESULT_CSV=%CD%\results\stress_test\hls_results.csv"
set "LENET_ACCURACY_THRESHOLD=0"
call "%VIVADO_HLS_ROOT%\bin\vivado_hls.bat" -f ..\level1\run_hls.tcl
if errorlevel 1 exit /b %errorlevel%
copy /y "..\level1\hls_work\lenet_level1_hls\solution1\csim\report\lenet_accel_csim.log" "results\stress_test\vivado_hls_2019_2_csim.log" >nul
python ..\level1\tools\lenet_validation.py compare ^
  --float-results results\stress_test\python_results.csv ^
  --hls-results results\stress_test\hls_results.csv ^
  --report results\stress_test\python_hls_comparison.json ^
  --mismatches results\stress_test\python_hls_mismatches.csv ^
  --threshold 0
if errorlevel 1 exit /b %errorlevel%
python tools\level2_validation.py finalize-report ^
  --comparison results\stress_test\python_hls_comparison.json ^
  --report results\stress_test\experiment_report.md
exit /b %errorlevel%
