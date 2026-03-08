@echo off
setlocal

set "VERSION="

:parse_args
if "%~1"=="" goto args_done

if /I "%~1"=="-v" (
  if "%~2"=="" (
    echo Error: Missing value after -v
    goto usage
  )
  set "VERSION=%~2"
  shift
  shift
  goto parse_args
)

if /I "%~1"=="--version" (
  if "%~2"=="" (
    echo Error: Missing value after --version
    goto usage
  )
  set "VERSION=%~2"
  shift
  shift
  goto parse_args
)

if /I "%~1"=="-h" goto usage
if /I "%~1"=="--help" goto usage

echo Error: Unknown argument "%~1"
goto usage

:args_done
if "%VERSION%"=="" (
  echo Error: Version is required.
  goto usage
)

set "PROJECT=src\NeuralSim.App\NeuralSim.App.csproj"
set "PUBLISH_DIR=src\NeuralSim.App\bin\Release\net9.0-windows\win-x64\publish"
set "SOURCE_EXE=%PUBLISH_DIR%\NeuralSim.App.exe"
set "TARGET_EXE=NeuralSim-v%VERSION%-win-x64.exe"

echo Publishing version %VERSION%...
dotnet publish "%PROJECT%" ^
  -c Release ^
  -r win-x64 ^
  --self-contained true ^
  -p:PublishSingleFile=true ^
  -p:IncludeNativeLibrariesForSelfExtract=true
if errorlevel 1 goto fail

if not exist "%SOURCE_EXE%" (
  echo Error: Source exe not found: %SOURCE_EXE%
  goto fail
)

if exist "%TARGET_EXE%" del /f /q "%TARGET_EXE%"
copy /y "%SOURCE_EXE%" "%TARGET_EXE%" >nul
if errorlevel 1 (
  echo Error: Failed to copy exe to %TARGET_EXE%
  goto fail
)

echo.
echo Output: %PUBLISH_DIR%\
echo Exe: %TARGET_EXE%
exit /b 0

:usage
echo.
echo Usage: %~nx0 -v VERSION
echo Example: %~nx0 -v 1.12
exit /b 1

:fail
echo.
echo Publish failed.
exit /b 1
