@echo off
setlocal

echo ========================================================
echo DeepSeek Quantization Framework - Maintenance Dashboard
echo ========================================================
echo.

:menu
echo Choose an option:
echo 1. Run cleanup script (remove cache files and temporary outputs)
echo 2. Organize notebooks
echo 3. Validate project organization
echo 4. Run inference tests
echo 5. Run full test suite
echo 6. Exit
echo.

set /p option="Enter option number: "

if "%option%"=="1" (
    echo.
    echo Running cleanup script...
    powershell -ExecutionPolicy Bypass -File "%~dp0cleanup_simple.ps1"
    goto end
)

if "%option%"=="2" (
    echo.
    echo Organizing notebooks...
    mkdir "%~dp0notebooks" 2>nul
    copy "%~dp0*.ipynb" "%~dp0notebooks\" /Y
    echo Notebooks copied to notebooks/ directory
    goto end
)

if "%option%"=="3" (
    echo.
    echo Validating project organization...
    python "%~dp0validate_organization.py"
    goto end
)

if "%option%"=="4" (
    echo.
    echo Running inference tests...
    powershell -ExecutionPolicy Bypass -File "%~dp0run_inference_tests.ps1"
    goto end
)

if "%option%"=="5" (
    echo.
    echo Running full test suite...
    powershell -ExecutionPolicy Bypass -File "%~dp0run_master_test.ps1"
    goto end
)

if "%option%"=="6" (
    goto exit
) else (
    echo.
    echo Invalid option. Please try again.
    echo.
    goto menu
)

:end
echo.
echo Operation completed.
echo.
pause
goto menu

:exit
echo.
echo Exiting maintenance dashboard.
endlocal
