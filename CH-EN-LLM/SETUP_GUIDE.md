# Setup Guide for LinguaBridge Local

This guide walks you through setting up and running LinguaBridge Local from scratch.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Python Environment Setup](#python-environment-setup)
3. [Installing Dependencies](#installing-dependencies)
4. [Obtaining Training Data](#obtaining-training-data)
5. [Running Each Phase](#running-each-phase)
6. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (ARM or x86/x64)
  - Recommended: Qualcomm Snapdragon X Elite or equivalent
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: 10GB+ free space
  - 2GB for models
  - 5GB for processed data
  - 3GB for training checkpoints

### Software Requirements
- **OS**: Windows 10/11 (ARM64 or x64)
- **Python**: 3.8, 3.9, or 3.10 (tested on 3.10)
- **PowerShell**: 5.1+ or PowerShell Core 7+

---

## Python Environment Setup

### Step 1: Check Python Installation

```powershell
python --version
```

Should output: `Python 3.8.x`, `3.9.x`, or `3.10.x`

If not installed:
1. Download from [python.org](https://www.python.org/downloads/)
2. During installation, check "Add Python to PATH"
3. Restart PowerShell

### Step 2: Create Virtual Environment

```powershell
# Navigate to project directory
cd C:\Users\noahg\Documents\CH-EN-LLM

# Create virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1
```

**Troubleshooting**: If you get an execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

You should see `(.venv)` in your prompt:
```
(.venv) PS C:\Users\noahg\Documents\CH-EN-LLM>
```

---

## Installing Dependencies

### Step 1: Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### Step 2: Install PaddlePaddle

**For ARM (Snapdragon X Elite)**:
```powershell
pip install paddlepaddle==2.6.0 -f https://www.paddlepaddle.org.cn/whl/windows/cpu-mkl-avx/stable.html
```

**For x86/x64**:
```powershell
pip install paddlepaddle==2.6.0
```

Verify installation:
```powershell
python -c "import paddle; print(paddle.__version__)"
```

### Step 3: Install All Dependencies

```powershell
pip install -r requirements.txt
```

This will install:
- PaddleNLP
- NumPy, Pandas
- jieba (Chinese tokenizer)
- sacremoses (English tokenizer)
- FastAPI, Uvicorn (for API)
- Other utilities

**Expected time**: 5-10 minutes depending on internet speed

### Step 4: Verify Installation

```powershell
python -c "import paddlenlp; print('PaddleNLP:', paddlenlp.__version__)"
python -c "import jieba; print('Jieba: OK')"
python -c "import sacremoses; print('Sacremoses: OK')"
```

All should print versions or "OK" without errors.

---

## Obtaining Training Data

You need parallel English-Chinese text files to train the model.

### Option 1: Sample Data (For Testing)

Create small test files manually:

```powershell
# Create data/raw directory if it doesn't exist
New-Item -ItemType Directory -Force -Path data/raw

# Create sample en.txt
@"
Hello, how are you?
This is a test sentence.
The weather is nice today.
I love learning languages.
"@ | Out-File -FilePath data/raw/en.txt -Encoding UTF8

# Create sample zh.txt
@"
你好，你怎么样？
这是一个测试句子。
今天天气很好。
我喜欢学习语言。
"@ | Out-File -FilePath data/raw/zh.txt -Encoding UTF8
```

### Option 2: Real Dataset (For Production)

Download a subset of WMT or CCMatrix dataset:

**WMT20 Chinese-English** (Recommended):
1. Visit: http://www.statmt.org/wmt20/translation-task.html
2. Download: `training-parallel-nc-v15.tgz`
3. Extract `news-commentary-v15.en-zh.en` and `.zh` files
4. Rename to `en.txt` and `zh.txt`
5. Place in `data/raw/`

**Or use this Python script** to download a filtered subset:

```python
# download_data.py
import requests
import gzip

def download_ccmatrix_sample(num_lines=10000):
    """Download sample from CCMatrix dataset."""
    # This is a simplified example
    # In practice, use Hugging Face datasets library
    
    print("Downloading sample data...")
    # Implement download logic here
    
    print(f"Downloaded {num_lines} sentence pairs")

if __name__ == "__main__":
    download_ccmatrix_sample()
```

### Data Format Requirements

- **Encoding**: UTF-8
- **Format**: One sentence per line
- **Alignment**: Line N in `en.txt` corresponds to line N in `zh.txt`
- **Size**: Minimum 1,000 pairs (testing), 100,000+ pairs (production)

---

## Running Each Phase

### Phase 1: Data Processing

```powershell
# Ensure virtual environment is activated
.\.venv\Scripts\Activate.ps1

# Run data processor
python -m src.data_processor
```

**Expected output**:
```
Loading raw data files...
Loaded 10000 sentence pairs
Cleaning text...
Cleaning: 10000/10000 (100.0%)
Tokenizing and filtering sentences...
Tokenizing: 10000/10000 (100.0%)
After filtering: 9523 valid sentence pairs
Building en vocabulary...
Final en vocabulary size: 15243
Building zh vocabulary...
Final zh vocabulary size: 12891
Saving processed data...
Data Processing Complete!
```

**Output files**:
- `data/processed/train_en.npy`
- `data/processed/train_zh.npy`
- `data/processed/val_en.npy`
- `data/processed/val_zh.npy`
- `data/processed/test_en.npy`
- `data/processed/test_zh.npy`
- `data/processed/en_word2idx.pkl`
- `data/processed/zh_word2idx.pkl`
- Plus idx2word files

**Time**: 1-5 minutes depending on dataset size

### Phase 2: Teacher Model Training (Cloud)

**⚠️ This requires a GPU server!**

Options:
1. **Alibaba Cloud ECS** with GPU instance
2. **Google Colab Pro+** with A100 GPU
3. **Skip this phase** and proceed to Phase 3 with random initialization (lower quality but runnable)

**On GPU server**:

```bash
# Upload code and data
# SSH into server
# Activate environment

python -m src.train_teacher
```

**Configuration**: Edit `config.yaml` before training:
```yaml
teacher:
  num_epochs: 3  # Reduce to 1 for quick testing
  batch_size: 8  # Reduce if OOM errors
  learning_rate: 5.0e-5
```

**Expected time**: 8-12 hours on V100 GPU for 100k pairs

**Output**:
- `models/teacher/best_model/` - Best checkpoint
- `models/teacher/final_model/` - Final checkpoint
- `data/soft_labels/soft_labels.h5` - For distillation

### Phase 3: Student Distillation (Local)

**This runs on your local ARM device!**

```powershell
python -m src.distill_local
```

**Configuration** (in `config.yaml`):
```yaml
student:
  num_epochs: 5  # Start with 1-2 for testing
  batch_size: 4  # Reduce to 2 if memory issues
  learning_rate: 3.0e-4
```

**Expected output**:
```
Starting Student Model Distillation Training
Loading vocabularies...
Initializing student model...
Loaded model: Qwen/Qwen2.5-0.5B
Student model parameters: 500.0M

Epoch 1/5
Training: 100%|████████| 2380/2380 [1:23:15<00:00]
Train - Total: 2.543, Soft: 1.234, Hard: 1.309
Validation: 100%|████████| 75/75 [02:15<00:00]
Val - Total: 2.187, Soft: 1.098, Hard: 1.089
Saved best model (val_loss: 2.187)
...
```

**Expected time**: 
- 1 epoch: ~1.5 hours on Snapdragon X Elite
- 5 epochs: ~8 hours

**Output**:
- `models/student/best_model/`
- `models/student/final_model/`
- `models/student/final_model/model.pdparams`

**Tip**: Run overnight or during non-working hours!

### Phase 4a: GUI Application

```powershell
python -m src.app_gui
```

**Expected behavior**:
1. Window opens showing "Loading model..."
2. After 10-30 seconds, status changes to "Ready to translate"
3. Enter English text in top box
4. Click "Translate"
5. Chinese translation appears in bottom box

**Screenshot of GUI**:
```
┌────────────────────────────────────────────┐
│ 🌉 LinguaBridge Local                      │
├────────────────────────────────────────────┤
│ English Input                              │
│ ┌────────────────────────────────────────┐ │
│ │ Hello, world!                          │ │
│ │                                        │ │
│ └────────────────────────────────────────┘ │
│  [Translate] [Clear] [Copy Translation]    │
├────────────────────────────────────────────┤
│ Chinese Translation                        │
│ ┌────────────────────────────────────────┐ │
│ │ 你好，世界！                             │ │
│ │                                        │ │
│ └────────────────────────────────────────┘ │
│ Ready to translate                         │
└────────────────────────────────────────────┘
```

### Phase 4b: REST API

```powershell
python -m src.app_api
```

**Expected output**:
```
INFO:     Started server process
INFO:     Waiting for application startup.
Starting LinguaBridge API...
Loading translation model...
Model loaded from models/student/final_model
Inference engine initialized
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Test the API**:

Open another PowerShell window:

```powershell
# Test health endpoint
curl http://127.0.0.1:8000/health

# Test translation (Windows curl syntax)
curl -X POST http://127.0.0.1:8000/translate `
  -H "Content-Type: application/json" `
  -d '{\"text\": \"Hello, world!\"}'
```

**Or use browser**:
- Navigate to: http://127.0.0.1:8000/docs
- Interactive Swagger UI for testing

---

## Troubleshooting

### Issue: "No module named 'paddle'"

**Solution**:
```powershell
# Ensure virtual environment is activated
.\.venv\Scripts\Activate.ps1

# Reinstall PaddlePaddle
pip install paddlepaddle==2.6.0
```

### Issue: "FileNotFoundError: Raw data files not found"

**Solution**:
```powershell
# Check files exist
Test-Path data/raw/en.txt
Test-Path data/raw/zh.txt

# If false, create sample data (see "Obtaining Training Data" section)
```

### Issue: Out of Memory (OOM) during training

**Solution**:
1. Reduce batch size in `config.yaml`:
   ```yaml
   student:
     batch_size: 2  # Down from 4
   ```

2. Enable gradient checkpointing (already enabled)

3. Close other applications

4. Use smaller dataset

### Issue: Model download fails (PaddleNLP)

**Solution**:
1. **Manual download**:
   - Visit: https://huggingface.co/Qwen/Qwen2.5-0.5B
   - Download all files
   - Place in: `models/qwen2.5-0.5b/`
   
2. **Update config.yaml**:
   ```yaml
   student:
     model_name: "models/qwen2.5-0.5b"  # Local path
   ```

### Issue: Slow inference on ARM

**Solution**:
1. Increase CPU threads:
   ```yaml
   hardware:
     num_threads: 6  # Up from 4
   ```

2. Enable caching (already enabled)

3. Use smaller input text (chunk long sentences)

### Issue: GUI doesn't open

**Solution**:
```powershell
# Check tkinter installation
python -c "import tkinter; print('OK')"

# If error, reinstall Python with Tcl/Tk support
# Or use API instead of GUI
```

### Issue: "ImportError: DLL load failed" (Windows)

**Solution**:
1. Install Visual C++ Redistributable:
   - Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Run installer

2. Restart PowerShell

---

## Next Steps

Once everything is running:

1. **Test with sample data**: Verify pipeline works end-to-end
2. **Get real data**: Download WMT or CCMatrix dataset
3. **Train teacher** (optional): On GPU server
4. **Train student**: On local ARM device
5. **Deploy**: Use GUI or API for translations

---

## Getting Help

- **Check logs**: `logs/linguabridge.log`
- **GitHub Issues**: Open issue with error message
- **Documentation**: Re-read README.md and PROJECT_SUMMARY.md

---

## Performance Benchmarks (Your Hardware)

After setup, benchmark your system:

```powershell
python -c "
from src.inference import TranslationInference
from src.utils import load_config
import time

config = load_config()
engine = TranslationInference(config)

text = 'This is a test sentence to measure translation speed.'

start = time.time()
translation = engine.translate(text)
end = time.time()

print(f'Translation: {translation}')
print(f'Time: {(end-start)*1000:.0f}ms')
"
```

**Expected times**:
- Snapdragon X Elite: 80-120ms
- Apple M1/M2: 60-100ms
- Intel i7 (laptop): 100-150ms

---

Good luck with your setup! 🚀
