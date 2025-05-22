# test_api.ps1
$ErrorActionPreference = "Stop"

# Activate virtual environment
. ".\.venv\Scripts\Activate.ps1"

# Install dependencies
pip install uvicorn httpx

# Start uvicorn server in background
$process = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-m uvicorn main:app --host 0.0.0.0 --port 8000" -PassThru -NoNewWindow

# Wait for server to start
Start-Sleep -Seconds 10

try {
    # Test predict endpoint
    $response = Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -Body '{"review":"Great movie!"}'
    Write-Output "API Response: $response"
    if ($response.sentiment -eq "positive") {
        Write-Output "API test passed"
    } else {
        Write-Output "API test failed: Unexpected sentiment"
        exit 1
    }
} catch {
    Write-Output "Error connecting to API: $_"
    exit 1
} finally {
    # Stop uvicorn process
    Stop-Process -Id $process.Id -Force
}