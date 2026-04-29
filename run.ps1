# Multi-process startup script
Write-Host "--- Starting Sign Language System ---" -ForegroundColor Cyan

# 1. Start Backend in a new window
Write-Host "Starting Backend (FastAPI)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; .\venv\Scripts\activate; uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

# 2. Start Frontend in a new window
Write-Host "Starting Frontend (React)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\frontend'; npm run dev"

Write-Host "------------------------------------" -ForegroundColor Green
Write-Host "The application should open in your browser shortly."
Write-Host "Backend: http://localhost:8000"
Write-Host "Frontend: http://localhost:5173"
Write-Host "------------------------------------"
