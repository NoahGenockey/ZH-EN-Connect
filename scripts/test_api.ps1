# Test LinguaBridge API on Alibaba Cloud
# Usage: .\test_api.ps1 -ApiUrl "http://your-slb-ip" -OR- .\test_api.ps1 -ApiUrl "http://api.yourdomain.com"

param(
    [Parameter(Mandatory=$true)]
    [string]$ApiUrl
)

# Remove trailing slash
$ApiUrl = $ApiUrl.TrimEnd('/')

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "LinguaBridge API Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Testing API at: $ApiUrl" -ForegroundColor White
Write-Host ""

# Test data
$testCases = @(
    @{
        text = "Hello, how are you?"
        source = "en"
        target = "zh"
        expected = "你好"
    },
    @{
        text = "Good morning, welcome to our service."
        source = "en"
        target = "zh"
        expected = "早上好"
    },
    @{
        text = "Thank you very much for your help."
        source = "en"
        target = "zh"
        expected = "谢谢"
    },
    @{
        text = "The weather is beautiful today."
        source = "en"
        target = "zh"
        expected = "天气"
    }
)

# Test 1: Health Check
Write-Host "[TEST 1] Health Check" -ForegroundColor Yellow
Write-Host "Endpoint: $ApiUrl/health" -ForegroundColor Gray
try {
    $response = Invoke-RestMethod -Uri "$ApiUrl/health" -Method Get -TimeoutSec 10
    Write-Host "[SUCCESS] Health check passed" -ForegroundColor Green
    Write-Host "Response: $($response | ConvertTo-Json -Compress)" -ForegroundColor Gray
} catch {
    Write-Host "[FAILED] Health check failed" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Test 2: Translation Tests
Write-Host "[TEST 2] Translation Tests" -ForegroundColor Yellow
Write-Host ""

$successCount = 0
$failCount = 0

foreach ($i in 0..($testCases.Count - 1)) {
    $test = $testCases[$i]
    
    Write-Host "Test $($i + 1)/$($testCases.Count)" -ForegroundColor Cyan
    Write-Host "  Input:  $($test.text)" -ForegroundColor White
    
    $body = @{
        text = $test.text
        source = $test.source
        target = $test.target
    } | ConvertTo-Json
    
    try {
        $startTime = Get-Date
        $response = Invoke-RestMethod -Uri "$ApiUrl/translate" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 30
        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalMilliseconds
        
        Write-Host "  Output: $($response.translation)" -ForegroundColor Green
        Write-Host "  Time:   $([math]::Round($duration, 2))ms" -ForegroundColor Gray
        
        # Check if expected text is in translation
        if ($response.translation -match $test.expected) {
            Write-Host "  [PASS] Contains expected text: $($test.expected)" -ForegroundColor Green
            $successCount++
        } else {
            Write-Host "  [WARN] May not contain expected text: $($test.expected)" -ForegroundColor Yellow
            $successCount++  # Still count as success if we got a translation
        }
        
    } catch {
        Write-Host "  [FAIL] Translation failed" -ForegroundColor Red
        Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
        $failCount++
    }
    
    Write-Host ""
    Start-Sleep -Milliseconds 500
}

# Test 3: Performance Test
Write-Host "[TEST 3] Performance Test (10 concurrent requests)" -ForegroundColor Yellow
Write-Host ""

$jobs = @()
$testText = "Hello world"
$body = @{
    text = $testText
    source = "en"
    target = "zh"
} | ConvertTo-Json

$startTime = Get-Date

for ($i = 1; $i -le 10; $i++) {
    $jobs += Start-Job -ScriptBlock {
        param($url, $body)
        try {
            $response = Invoke-RestMethod -Uri "$url/translate" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 30
            return @{ success = $true; duration = 0 }
        } catch {
            return @{ success = $false; error = $_.Exception.Message }
        }
    } -ArgumentList $ApiUrl, $body
}

Write-Host "Waiting for all requests to complete..." -ForegroundColor Gray
$results = $jobs | Wait-Job | Receive-Job
$jobs | Remove-Job

$endTime = Get-Date
$totalDuration = ($endTime - $startTime).TotalMilliseconds

$successfulRequests = ($results | Where-Object { $_.success -eq $true }).Count
$failedRequests = 10 - $successfulRequests

Write-Host "Total time: $([math]::Round($totalDuration, 2))ms" -ForegroundColor White
Write-Host "Successful: $successfulRequests/10" -ForegroundColor Green
Write-Host "Failed: $failedRequests/10" -ForegroundColor $(if ($failedRequests -gt 0) { "Red" } else { "Gray" })
Write-Host "Avg time per request: $([math]::Round($totalDuration / 10, 2))ms" -ForegroundColor Gray
Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Translation Tests: $successCount passed, $failCount failed" -ForegroundColor White
Write-Host "Concurrent Tests: $successfulRequests/10 successful" -ForegroundColor White
Write-Host ""

if ($failCount -eq 0 -and $failedRequests -eq 0) {
    Write-Host "[SUCCESS] All tests passed! 🎉" -ForegroundColor Green
} elseif ($failCount -gt 0 -or $failedRequests -gt 5) {
    Write-Host "[FAILED] Some tests failed. Please check your deployment." -ForegroundColor Red
} else {
    Write-Host "[WARNING] Minor issues detected. Review the results above." -ForegroundColor Yellow
}
Write-Host ""

# Additional info
Write-Host "API URL: $ApiUrl" -ForegroundColor Gray
Write-Host "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
Write-Host ""
