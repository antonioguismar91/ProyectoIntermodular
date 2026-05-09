@echo off
setlocal
cd /d "%~dp0"

echo ==============================================
echo  GreenLight AI
echo  Sistema DSS para Prevencion de Lesiones
echo  Iniciando aplicacion Streamlit
echo ==============================================

echo.
echo Comprobando entorno virtual...
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: El entorno virtual no fue encontrado.
    echo Por favor, ejecuta primero setup_env.bat
    pause
    exit /b 1
)

echo.
echo Comprobando configuracion de Mistral...
if exist ".streamlit\secrets.toml" (
    echo Archivo secrets.toml encontrado. Modulo NLP activo.
) else (
    echo AVISO: No se encontro .streamlit\secrets.toml
    echo El modulo NLP funcionara en modo heuristico.
    echo Ejecuta setup_env.bat para configurar la API key.
)

echo.
echo Iniciando GreenLight AI...
echo Abriendo en: http://localhost:8501
echo Para detener la aplicacion pulsa Ctrl+C en esta ventana.
echo.
call ".venv\Scripts\python.exe" -m streamlit run ui\app.py

if errorlevel 1 (
    echo.
    echo ERROR: La aplicacion no pudo iniciarse.
    echo Asegurate de que ui\app.py existe y las dependencias estan instaladas.
    pause
    exit /b 1
)
