@echo off
title SkillGraph Backend Server
cd /d "%~dp0"
echo.
echo  ============================================
echo   SkillGraph ^& Student Growth Intelligence
echo   Starting backend on http://localhost:8000
echo  ============================================
echo.
echo  Open your browser at: http://localhost:8000
echo  API docs at:          http://localhost:8000/docs
echo.
echo  Press Ctrl+C to stop the server.
echo.
python -m uvicorn hackathon:app --reload --host 0.0.0.0 --port 8000
pause
