@echo off
echo ========================================
echo   Banking RAG Chatbot - Starting Up
echo ========================================

echo.
echo [1/3] Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo [2/3] Starting Backend API (FastAPI)...
start "Backend" cmd /k "cd /d E:\assestment\banking-rag-chatbot && python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"

echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo.
echo [3/3] Starting Frontend (Streamlit)...
start "Frontend" cmd /k "cd /d E:\assestment\banking-rag-chatbot && python -m streamlit run frontend/app.py --server.port 8501"

echo.
echo ========================================
echo   App is starting!
echo   Backend:  http://localhost:8000/docs
echo   Frontend: http://localhost:8501
echo ========================================
echo.
echo Login: demo@bankingchatbot.com / Demo@1234
pause
