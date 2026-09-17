# Run QED-Net 2.0 live: backend :8000 + dashboard :5173
# Usage: powershell -ExecutionPolicy Bypass -File scripts/dev.ps1
Start-Process powershell -ArgumentList "-NoExit","-Command","Set-Location '$PSScriptRoot/..'; python -m uvicorn api.main:app --host 127.0.0.1 --port 8000"
Start-Process powershell -ArgumentList "-NoExit","-Command","Set-Location '$PSScriptRoot/../frontend'; npm run dev -- --host 127.0.0.1 --port 5173"
Start-Sleep 3
Start-Process "http://127.0.0.1:5173"
Write-Host "API: http://127.0.0.1:8000/health  UI: http://127.0.0.1:5173"
