# test_api.ps1
$ErrorActionPreference = "Stop"

# Activate virtual environment
. ".\.venv\Scripts\Activate.ps1"

# Install dependencies
pip install uvicorn httpx

# Check if port 8000 is in use
$portInUse = netstat -aon | Select-String ":8000" | Select-String "LISTENING"
if ($portInUse) {
    Write-Output "Port 8000 is in use. Attempting to free it..."
    $processId = ($portInUse -split '\s+')[-1]
    try {
        Stop-Process -Id $processId -Force -ErrorAction Stop
        Write-Output "Terminated process $processId on port 8000"
        Start-Sleep -Seconds 2
    } catch {
        Write-Output "Failed to terminate process $processId : $($_.Exception.Message)"
        exit 1
    }
}

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
    Write-Output "Error connecting to API: $($_.Exception.Message)"
    exit 1
} finally {
    # Stop uvicorn process
    if ($process -and -not $process.HasExited) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        Write-Output "Stopped uvicorn process"
    } else {
        Write-Output "No uvicorn process to stop"
    }
}
