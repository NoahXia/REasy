@echo off
setlocal
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
set "GDEFLATE_DLL=.cache\gdeflate\libGDeflate.dll"
set "PYTHONNOUSERSITE=1"
set "REASY_ORIGINAL_PATH=%PATH%"
set "PROJECTS_DIR=%CD%\dist\projects"
set "PROJECTS_BACKUP=%CD%\.cache\dist-projects-backup"
set "SETTINGS_FILE=%CD%\dist\settings.json"
set "SETTINGS_BACKUP=%CD%\.cache\dist-settings-backup.json"
set "PROJECTS_SAVED=0"
set "SETTINGS_SAVED=0"

rem Preserve user data because the normal build replaces the entire dist tree.
rem Keep each backup after a successful restore so a failed future build remains recoverable.
if exist "%PROJECTS_DIR%" (
  if exist "%PROJECTS_BACKUP%" rmdir /S /Q "%PROJECTS_BACKUP%" || goto :build_failed
  xcopy /E /I /H /Y "%PROJECTS_DIR%" "%PROJECTS_BACKUP%\" >nul
  if errorlevel 2 goto :build_failed
  set "PROJECTS_SAVED=1"
  echo Backed up dist\projects to .cache\dist-projects-backup
)
if exist "%SETTINGS_FILE%" (
  copy /Y "%SETTINGS_FILE%" "%SETTINGS_BACKUP%" >nul || goto :build_failed
  set "SETTINGS_SAVED=1"
  echo Backed up dist\settings.json to .cache\dist-settings-backup.json
)

if exist build rmdir /S /Q build || goto :build_failed
if exist dist rmdir /S /Q dist || goto :build_failed

if /I "%GITHUB_ACTIONS%"=="true" (
  set "PY=python"
  call "%~dp0prepare_env.bat" -UseCurrentPython || goto :build_failed
) else (
  set "PATH=%CD%\.venv\Scripts;%PATH%"
  call "%~dp0prepare_env.bat" || goto :build_failed
)

rem Keep unrelated Qt/ICU installations on the machine out of PyInstaller's
rem dependency resolution. Qt uses the Windows system ICU on supported hosts.
set "PATH=%CD%\.venv\Scripts;%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem"

"%PY%" -c "import runpy, sys, sysconfig; sys.path.insert(0, sysconfig.get_path('stdlib')); runpy.run_module('PyInstaller', run_name='__main__')" --onefile --windowed --icon=resources/icons/reasy_editor_logo.ico --version-file=version.txt ^
  --runtime-hook scripts\pyi_rth_preload_qt.py ^
  --collect-submodules file_handlers ^
  --hidden-import fast_pakresolve --collect-binaries fast_pakresolve ^
  --hidden-import fast_string_scan --collect-binaries fast_string_scan ^
  --hidden-import fastmesh --collect-binaries fastmesh ^
  --hidden-import texture2ddecoder --collect-all texture2ddecoder ^
  --add-binary "%GDEFLATE_DLL%;tools\runtimes\win-x64\native" ^
  REasy.py || goto :build_failed

set "PATH=%REASY_ORIGINAL_PATH%"

xcopy /E /I /Y resources dist\resources || goto :build_failed
if exist dist\resources\data\dumps rmdir /S /Q dist\resources\data\dumps
if exist dist\resources\patches rmdir /S /Q dist\resources\patches
if not exist dist\resources\i18n mkdir dist\resources\i18n
xcopy /Y /I resources\i18n\ dist\resources\i18n\ || goto :build_failed
copy "resources\images\reasy_guy.png" "dist\resources\images\reasy_guy.png" || goto :build_failed
if not exist dist\resources\scripts mkdir dist\resources\scripts
copy "scripts\auto_update.ps1" "dist\resources\scripts\auto_update.ps1" || goto :build_failed
copy "resources\data\dumps\*.json" dist\ || goto :build_failed

call :restore_user_data || exit /b 1

if not exist dist\REasy.exe exit /b 1
echo Built dist\REasy.exe
exit /b 0

:restore_user_data
if "%PROJECTS_SAVED%"=="1" (
  xcopy /E /I /H /Y "%PROJECTS_BACKUP%" "%PROJECTS_DIR%\" >nul
  if errorlevel 2 exit /b 1
  echo Restored dist\projects from .cache\dist-projects-backup
)
if "%SETTINGS_SAVED%"=="1" (
  if not exist "%CD%\dist" mkdir "%CD%\dist" || exit /b 1
  copy /Y "%SETTINGS_BACKUP%" "%SETTINGS_FILE%" >nul || exit /b 1
  echo Restored dist\settings.json from .cache\dist-settings-backup.json
)
exit /b 0

:build_failed
set "PATH=%REASY_ORIGINAL_PATH%"
echo Build failed. Restoring preserved user data...
call :restore_user_data
exit /b 1
