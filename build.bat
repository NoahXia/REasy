@echo off
setlocal
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
set "GDEFLATE_DLL=.cache\gdeflate\libGDeflate.dll"
set "PYTHONNOUSERSITE=1"
set "PROJECTS_DIR=%CD%\dist\projects"
set "PROJECTS_BACKUP=%CD%\.cache\dist-projects-backup"

rem Preserve user projects because the normal build replaces the entire dist tree.
rem Keep the backup after a successful restore so a failed future build remains recoverable.
if exist "%PROJECTS_DIR%" (
  if exist "%PROJECTS_BACKUP%" rmdir /S /Q "%PROJECTS_BACKUP%" || exit /b 1
  xcopy /E /I /H /Y "%PROJECTS_DIR%" "%PROJECTS_BACKUP%\" >nul
  if errorlevel 2 exit /b 1
  echo Backed up dist\projects to .cache\dist-projects-backup
)

if exist build rmdir /S /Q build || exit /b 1
if exist dist rmdir /S /Q dist || exit /b 1

if /I "%GITHUB_ACTIONS%"=="true" (
  set "PY=python"
  call "%~dp0prepare_env.bat" -UseCurrentPython || exit /b 1
) else (
  set "PATH=%CD%\.venv\Scripts;%PATH%"
  call "%~dp0prepare_env.bat" || exit /b 1
)

rem Keep unrelated Qt/ICU installations on the machine out of PyInstaller's
rem dependency resolution. Qt uses the Windows system ICU on supported hosts.
set "REASY_ORIGINAL_PATH=%PATH%"
set "PATH=%CD%\.venv\Scripts;%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem"

"%PY%" -c "import runpy, sys, sysconfig; sys.path.insert(0, sysconfig.get_path('stdlib')); runpy.run_module('PyInstaller', run_name='__main__')" --onefile --windowed --icon=resources/icons/reasy_editor_logo.ico --version-file=version.txt ^
  --runtime-hook scripts\pyi_rth_preload_qt.py ^
  --collect-submodules file_handlers ^
  --hidden-import fast_pakresolve --collect-binaries fast_pakresolve ^
  --hidden-import fast_string_scan --collect-binaries fast_string_scan ^
  --hidden-import fastmesh --collect-binaries fastmesh ^
  --hidden-import texture2ddecoder --collect-all texture2ddecoder ^
  --add-binary "%GDEFLATE_DLL%;tools\runtimes\win-x64\native" ^
  REasy.py || exit /b 1

set "PATH=%REASY_ORIGINAL_PATH%"

xcopy /E /I /Y resources dist\resources || exit /b 1
if exist dist\resources\data\dumps rmdir /S /Q dist\resources\data\dumps
if exist dist\resources\patches rmdir /S /Q dist\resources\patches
if not exist dist\resources\i18n mkdir dist\resources\i18n
xcopy /Y /I resources\i18n\ dist\resources\i18n\ || exit /b 1
copy "resources\images\reasy_guy.png" "dist\resources\images\reasy_guy.png" || exit /b 1
if not exist dist\resources\scripts mkdir dist\resources\scripts
copy "scripts\auto_update.ps1" "dist\resources\scripts\auto_update.ps1" || exit /b 1
copy "resources\data\dumps\*.json" dist\ || exit /b 1

if exist "%PROJECTS_BACKUP%" (
  xcopy /E /I /H /Y "%PROJECTS_BACKUP%" "%PROJECTS_DIR%\" >nul
  if errorlevel 2 exit /b 1
  echo Restored dist\projects from .cache\dist-projects-backup
)

if not exist dist\REasy.exe exit /b 1
echo Built dist\REasy.exe
