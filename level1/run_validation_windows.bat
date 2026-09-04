@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not defined VIVADO_HLS_ROOT set "VIVADO_HLS_ROOT=E:\use\cpu\Vivado\2019.2"
set "MNIST_IMAGES=data\mnist\t10k-images-idx3-ubyte"
set "MNIST_LABELS=data\mnist\t10k-labels-idx1-ubyte"
set "BATCH_BLOB=data\lenet_accuracy_10000.bin"
set "RESULT_DIR=results\windows_2019_2"

if not exist "%VIVADO_HLS_ROOT%\settings64.bat" (
  echo ERROR: Vivado HLS settings not found: %VIVADO_HLS_ROOT%\settings64.bat
  echo Set VIVADO_HLS_ROOT to the Vivado 2019.2 installation directory.
  exit /b 2
)
if not exist "%MNIST_IMAGES%" (
  echo ERROR: Missing %MNIST_IMAGES%
  exit /b 2
)
if not exist "%MNIST_LABELS%" (
  echo ERROR: Missing %MNIST_LABELS%
  exit /b 2
)
if not exist "data\lenet_accuracy_1.bin" (
  echo ERROR: Missing data\lenet_accuracy_1.bin
  exit /b 2
)

where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: python is not available in PATH.
  exit /b 2
)

if not exist "%RESULT_DIR%" mkdir "%RESULT_DIR%"

echo [1/6] Validate official MNIST files
python tools\lenet_validation.py validate-mnist ^
  --images "%MNIST_IMAGES%" ^
  --labels "%MNIST_LABELS%" ^
  --json-output "%RESULT_DIR%\mnist_summary.json"
if errorlevel 1 exit /b %errorlevel%

echo [2/6] Build the 10000-sample HLS input blob
python tools\lenet_validation.py make-blob ^
  --parameters data\lenet_accuracy_1.bin ^
  --images "%MNIST_IMAGES%" ^
  --labels "%MNIST_LABELS%" ^
  --output "%BATCH_BLOB%" ^
  --metadata "%RESULT_DIR%\blob_metadata.json"
if errorlevel 1 exit /b %errorlevel%

echo [3/6] Run the Python float32 reference
python tools\lenet_validation.py run-float ^
  --blob "%BATCH_BLOB%" ^
  --results "%RESULT_DIR%\float_results.csv" ^
  --summary "%RESULT_DIR%\float_summary.json" ^
  --expected-count 10000
if errorlevel 1 exit /b %errorlevel%

echo [4/6] Run Vivado HLS 2019.2 C simulation
call "%VIVADO_HLS_ROOT%\settings64.bat"
set "PATH=%VIVADO_HLS_ROOT%\tps\win64\msys64\mingw64\bin;%VIVADO_HLS_ROOT%\tps\mingw\6.2.0\win64.o\nt\bin;%PATH%"
set "LENET_SKIP_SYNTH=1"
set "LENET_ACCURACY_BLOB=%CD%\%BATCH_BLOB%"
set "LENET_RESULT_CSV=%CD%\%RESULT_DIR%\hls_results.csv"
call "%VIVADO_HLS_ROOT%\bin\vivado_hls.bat" -f run_hls.tcl
if errorlevel 1 exit /b %errorlevel%

echo [5/6] Compare Python and HLS predictions
python tools\lenet_validation.py compare ^
  --float-results "%RESULT_DIR%\float_results.csv" ^
  --hls-results "%RESULT_DIR%\hls_results.csv" ^
  --report "%RESULT_DIR%\validation_report.json" ^
  --mismatches "%RESULT_DIR%\mismatches.csv" ^
  --threshold 90
if errorlevel 1 exit /b %errorlevel%

echo [6/6] Complete
echo Results: %CD%\%RESULT_DIR%
exit /b 0
