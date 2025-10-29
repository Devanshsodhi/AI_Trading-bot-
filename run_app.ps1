# AI Trading System Launcher
# Quick launcher for the Streamlit app

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   🤖 AI Trading System Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "⚠️  WARNING: .env file not found!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Creating .env file from template..." -ForegroundColor Yellow
    
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "✅ Created .env file" -ForegroundColor Green
        Write-Host ""
        Write-Host "📝 IMPORTANT: Edit .env file and add your GROQ_API_KEY" -ForegroundColor Yellow
        Write-Host "   Get free key from: https://console.groq.com/" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Press any key to continue (or Ctrl+C to exit and edit .env)..." -ForegroundColor Yellow
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    } else {
        Write-Host "❌ .env.example not found. Please create .env file manually." -ForegroundColor Red
        Write-Host ""
        exit 1
    }
}

# Check if requirements are installed
Write-Host "🔍 Checking dependencies..." -ForegroundColor Cyan
try {
    python -c "import streamlit" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠️  Streamlit not found. Installing dependencies..." -ForegroundColor Yellow
        Write-Host ""
        pip install -r requirements.txt
        Write-Host ""
    }
} catch {
    Write-Host "⚠️  Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
    Write-Host ""
}

Write-Host "🚀 Starting AI Trading System..." -ForegroundColor Green
Write-Host ""
Write-Host "📊 Dashboard will open in your browser at:" -ForegroundColor Cyan
Write-Host "   http://localhost:8501" -ForegroundColor White
Write-Host ""
Write-Host "💡 Tips:" -ForegroundColor Yellow
Write-Host "   - Use 🎯 Balanced mode for best results" -ForegroundColor White
Write-Host "   - First analysis takes longer (training models)" -ForegroundColor White
Write-Host "   - Subsequent analyses are much faster (cached)" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Launch Streamlit
streamlit run streamlit_app.py
