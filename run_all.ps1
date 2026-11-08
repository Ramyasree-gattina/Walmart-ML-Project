Write-Host "========================================" -ForegroundColor Cyan
Write-Host " 1. RUNNING PYTEST UNIT TESTS..." -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
python -m pytest

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " 2. SYNCING AND PUSHING TO GITHUB..." -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
git add .
git commit -m "Auto test and sync for Lab 13"
git push origin main

Write-Host "`n========================================" -ForegroundColor Green
Write-Host " SUCCESS! LAB 13 READY FOR EVALUATION ✅" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green