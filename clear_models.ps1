# Clear all TimeGAN models to force retraining with new config
Write-Host "Clearing old TimeGAN models..." -ForegroundColor Yellow
Remove-Item -Path "models\timegan_*.pth" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "models\timegan_*.pkl" -Force -ErrorAction SilentlyContinue
Write-Host "✅ Old models cleared! Next analysis will retrain with 50 epochs." -ForegroundColor Green
