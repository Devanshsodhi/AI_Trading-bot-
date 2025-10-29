@echo off
REM AI Trading System Launcher (Batch version)

echo.
echo ========================================
echo    AI Trading System Launcher
echo ========================================
echo.

REM Check if .env exists
if not exist .env (
    echo WARNING: .env file not found!
    echo.
    if exist .env.example (
        echo Creating .env from template...
        copy .env.example .env
        echo.
        echo IMPORTANT: Edit .env file and add your GROQ_API_KEY
        echo Get free key from: https://console.groq.com/
        echo.
        pause
    ) else (
        echo ERROR: .env.example not found!
        pause
        exit /b 1
    )
)

echo Starting AI Trading System...
echo.
echo Dashboard will open at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.
echo ========================================
echo.

streamlit run streamlit_app.py
