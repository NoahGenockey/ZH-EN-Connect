# Upload LinguaBridge Models to Alibaba Cloud OSS
# Usage: .\upload_to_oss.ps1 -BucketName "linguabridge-models-xxxxx"

param(
    [Parameter(Mandatory=$true)]
    [string]$BucketName,
    
    [Parameter(Mandatory=$false)]
    [string]$ProjectPath = "C:\Users\noahg\Documents\CH-EN-LLM",
    
    [Parameter(Mandatory=$false)]
    [switch]$FullUpload = $false
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "LinguaBridge - OSS Upload Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if ossutil is installed
if (-not (Get-Command "ossutil" -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] ossutil not found!" -ForegroundColor Red
    Write-Host "Please install ossutil from: https://www.alibabacloud.com/help/oss/developer-reference/ossutil" -ForegroundColor Yellow
    Write-Host "After installing, run: ossutil config" -ForegroundColor Yellow
    exit 1
}

# Change to project directory
if (-not (Test-Path $ProjectPath)) {
    Write-Host "[ERROR] Project path not found: $ProjectPath" -ForegroundColor Red
    exit 1
}

Set-Location $ProjectPath
Write-Host "[INFO] Working directory: $ProjectPath" -ForegroundColor Green
Write-Host ""

# Test OSS connection
Write-Host "[INFO] Testing OSS connection..." -ForegroundColor Green
$testResult = ossutil ls "oss://$BucketName" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Cannot connect to OSS bucket: $BucketName" -ForegroundColor Red
    Write-Host "Please check:" -ForegroundColor Yellow
    Write-Host "1. Bucket name is correct" -ForegroundColor Yellow
    Write-Host "2. ossutil is configured: ossutil config" -ForegroundColor Yellow
    Write-Host "3. You have access to this bucket" -ForegroundColor Yellow
    exit 1
}
Write-Host "[SUCCESS] Connected to OSS bucket" -ForegroundColor Green
Write-Host ""

# Function to upload with progress
function Upload-ToOSS {
    param(
        [string]$LocalPath,
        [string]$OSSPath,
        [string]$Description
    )
    
    if (-not (Test-Path $LocalPath)) {
        Write-Host "[SKIP] $Description - Path not found: $LocalPath" -ForegroundColor Yellow
        return
    }
    
    Write-Host "[UPLOADING] $Description..." -ForegroundColor Cyan
    Write-Host "  From: $LocalPath" -ForegroundColor Gray
    Write-Host "  To:   oss://$BucketName/$OSSPath" -ForegroundColor Gray
    
    $result = ossutil cp -r $LocalPath "oss://$BucketName/$OSSPath" --update 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[SUCCESS] $Description uploaded" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Failed to upload $Description" -ForegroundColor Red
        Write-Host $result -ForegroundColor Red
    }
    Write-Host ""
}

# Upload strategy
if ($FullUpload) {
    Write-Host "[MODE] Full upload (all files)" -ForegroundColor Magenta
    Write-Host ""
    
    # Upload everything
    Upload-ToOSS -LocalPath "data\raw\" -OSSPath "data/raw/" -Description "Raw training data"
    Upload-ToOSS -LocalPath "data\processed\" -OSSPath "data/processed/" -Description "Processed data"
    Upload-ToOSS -LocalPath "data\soft_labels\" -OSSPath "data/soft_labels/" -Description "Soft labels"
    Upload-ToOSS -LocalPath "models\teacher\" -OSSPath "models/teacher/" -Description "Teacher model"
    Upload-ToOSS -LocalPath "models\student\" -OSSPath "models/student/" -Description "Student model"
    Upload-ToOSS -LocalPath "config.yaml" -OSSPath "config.yaml" -Description "Configuration file"
    Upload-ToOSS -LocalPath "requirements.txt" -OSSPath "requirements.txt" -Description "Requirements file"
    
} else {
    Write-Host "[MODE] Production upload (models and config only)" -ForegroundColor Magenta
    Write-Host ""
    
    # Essential files for deployment
    Upload-ToOSS -LocalPath "models\student\" -OSSPath "models/student/" -Description "Student model (required)"
    Upload-ToOSS -LocalPath "data\processed\" -OSSPath "data/processed/" -Description "Processed data (vocabularies)"
    Upload-ToOSS -LocalPath "config.yaml" -OSSPath "config.yaml" -Description "Configuration file"
    
    # Optional: teacher model
    if (Test-Path "models\teacher\") {
        $uploadTeacher = Read-Host "Upload teacher model? (y/n) [n]"
        if ($uploadTeacher -eq "y") {
            Upload-ToOSS -LocalPath "models\teacher\" -OSSPath "models/teacher/" -Description "Teacher model"
        }
    }
}

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Upload Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Bucket: oss://$BucketName" -ForegroundColor White
Write-Host ""
Write-Host "To view uploaded files:" -ForegroundColor Yellow
Write-Host "  ossutil ls oss://$BucketName/ -r" -ForegroundColor Gray
Write-Host ""
Write-Host "To download on ECS instance:" -ForegroundColor Yellow
Write-Host "  ossutil cp -r oss://$BucketName/models/student/ /opt/linguabridge/models/student/" -ForegroundColor Gray
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. SSH to your ECS instance" -ForegroundColor White
Write-Host "2. Run the setup script: bash deploy_setup_alibaba.sh" -ForegroundColor White
Write-Host "3. Download models from OSS" -ForegroundColor White
Write-Host "4. Start the API service" -ForegroundColor White
Write-Host ""
