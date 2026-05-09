@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ==============================================
echo  GreenLight AI
echo  Sistema DSS para Prevencion de Lesiones
echo  Instalacion de entorno y dependencias
echo ==============================================

echo.
echo [1/5] Comprobando instalacion de Python...
py -3.10 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.10 no encontrado con el launcher "py".
    echo Por favor, instala Python 3.10.x desde https://www.python.org
    echo y asegurate de que el Python Launcher este disponible.
    pause
    exit /b 1
)
echo Python 3.10 encontrado correctamente.

echo.
echo [2/5] Creando entorno virtual...
if not exist ".venv" (
    py -3.10 -m venv .venv
    echo Entorno virtual creado en .venv
) else (
    echo El entorno virtual ya existe. Reutilizando .venv
)

echo.
echo [3/5] Actualizando pip...
call ".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: No se pudo actualizar pip.
    pause
    exit /b 1
)

echo.
echo [4/5] Instalando dependencias del proyecto...
call ".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: No se pudieron instalar las dependencias.
    echo Revisa requirements.txt y tu conexion a internet.
    pause
    exit /b 1
)

echo.
echo [5/5] Configurando API key de Mistral (secrets.toml)...
if not exist ".streamlit" (
    mkdir .streamlit
)

if not exist ".streamlit\secrets.toml" (
    echo.
    echo El modulo NLP de GreenLight AI usa la API de Mistral AI.
    echo Puedes obtener una API key gratuita en: https://console.mistral.ai
    echo (Plan Experiment - sin coste)
    echo.
    set /p MISTRAL_KEY="Introduce tu MISTRAL_API_KEY (ENTER para omitir): "

    if not "!MISTRAL_KEY!"=="" (
        echo MISTRAL_API_KEY = "!MISTRAL_KEY!"> .streamlit\secrets.toml
        echo API key guardada correctamente en .streamlit\secrets.toml
        echo Este archivo esta excluido del repositorio por .gitignore
    ) else (
        echo MISTRAL_API_KEY = ""> .streamlit\secrets.toml
        echo Sin API key. El modulo NLP funcionara en modo heuristico.
    )
) else (
    echo Archivo .streamlit\secrets.toml ya existe. No se sobreescribe.
    echo Si necesitas cambiar la API key, edita el archivo manualmente.
)

echo.
echo ==============================================
echo  Instalacion completada con exito.
echo  Ejecuta run_app.bat para iniciar GreenLight AI
echo ==============================================
pause
