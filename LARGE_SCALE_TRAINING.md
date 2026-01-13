# Large-Scale Training Guide (71M+ Sentences)

This guide covers training LinguaBridge on massive datasets (71 million sentence pairs) using Alibaba Cloud.

## ⚠️ Reality Check

Training on 71 million sentences is **enterprise-scale**. Here are the realistic numbers:

### Time Estimates

| Configuration | GPU | Teacher Training | Student Training | Total Time | Total Cost |
|---------------|-----|------------------|------------------|------------|------------|
| **Full Dataset (71M)** | | | | | |
| A10 (24GB) | 1× A10 | 38-50 days | 12-18 days | 50-68 days | ¥36,000-49,000 |
| A100 (40GB) | 1× A100 | 18-25 days | 8-12 days | 26-37 days | ¥31,000-44,000 |
| A100 4-GPU | 4× A100 | 5-7 days | 2-3 days | 7-10 days | ¥25,000-36,000 |
| **Sampled Dataset (5M)** | | | | | |
| A10 (24GB) | 1× A10 | 2.5-3.5 days | 1-1.5 days | 3.5-5 days | ¥2,500-3,600 |
| A100 (40GB) | 1× A100 | 1.5-2 days | 0.5-1 day | 2-3 days | ¥2,400-3,600 |
| **Sampled Dataset (10M)** | | | | | |
| A10 (24GB) | 1× A10 | 5-7 days | 2-3 days | 7-10 days | ¥5,000-7,200 |
| A100 (40GB) | 1× A100 | 3-4 days | 1-2 days | 4-6 days | ¥4,800-7,200 |

### Cost per Hour (Alibaba Cloud GPU Instances)

| Instance Type | GPU | vCPU | RAM | Cost/Hour | Daily Cost |
|---------------|-----|------|-----|-----------|------------|
| ecs.gn7i-c16g1.4xlarge | 1× A10 (24GB) | 16 | 60GB | ¥30 | ¥720 |
| ecs.gn7i-c32g1.8xlarge | 1× A100 (40GB) | 32 | 188GB | ¥50 | ¥1,200 |
| ecs.gn7i-c64g1.16xlarge | 4× A10 (96GB) | 64 | 240GB | ¥120 | ¥2,880 |

---

## Recommended Strategies

### Strategy 1: Intelligent Sampling (Recommended for Most Use Cases)

**Best for**: Production systems with limited budget, still excellent quality

**Approach**: Use 5-10M highest-quality sentence pairs

**Benefits**:
- 95%+ of the translation quality at 7-14% of the cost
- 10x faster training
- Modern LLMs learn efficiently from smaller, high-quality data
- Easier to debug and iterate

**How to implement**:

1. **Edit config.yaml**:
```yaml
data:
  sample_size: 5000000  # 5 million sentences
  sample_strategy: "quality_filtered"  # Or "diverse" for variety
  quality_threshold: 0.8  # Only keep high-quality pairs
```

2. **Quality filtering criteria**:
   - Length ratio (EN/ZH between 0.6-1.8)
   - No repeated tokens
   - Proper punctuation
   - No encoding errors
   - Vocabulary richness

3. **Run processing**:
```bash
python run.py process
```

**Expected results**:
- Training time: 3-5 days on A10
- Cost: ¥2,500-3,600 (~$350-500)
- Model quality: 93-97% of full dataset quality
- BLEU score: 35-40 (excellent for general translation)

---

### Strategy 2: Reduced Epochs (Full Dataset)

**Best for**: When you need to use the entire dataset

**Approach**: Train for 1 epoch instead of 3

**Edit config.yaml**:
```yaml
teacher:
  num_epochs: 1  # Changed from 3
  batch_size: 16  # Increase if GPU allows
  gradient_accumulation_steps: 2  # Effective batch: 32
  
  # Add these for large datasets:
  save_steps: 10000  # Checkpoint every 10k steps
  eval_steps: 5000   # Evaluate every 5k steps
  max_steps: 500000  # Optional: cap total steps
```

**Expected results**:
- Training time: 13-17 days on A10, 6-9 days on A100
- Cost: ¥9,360-12,240 (~$1,300-1,700)
- Model quality: 90-95% of 3-epoch quality
- One epoch over 71M is still **massive** exposure

---

### Strategy 3: Multi-GPU Parallel Training

**Best for**: Need full dataset + fast turnaround

**Approach**: Use data parallelism with 4 GPUs

**Alibaba Cloud Instance**: ecs.gn7i-c64g1.16xlarge (4× A10)

**Benefits**:
- Near-linear speedup (3.5-4x faster)
- Train full 71M dataset in 7-10 days
- Worth it for time-sensitive projects

**Implementation**:

1. **Modify training script for distributed training**:
```python
# In train_teacher.py, add distributed setup
import paddle.distributed as dist

# Initialize distributed training
dist.init_parallel_env()

# Wrap model in DataParallel
model = paddle.DataParallel(model)

# Distributed sampler
train_sampler = paddle.io.DistributedBatchSampler(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
```

2. **Launch with multiple GPUs**:
```bash
# On Alibaba Cloud 4-GPU instance
python -m paddle.distributed.launch \
    --gpus 0,1,2,3 \
    src/train_teacher.py
```

**Expected results**:
- Training time: 9-12 days (full dataset, 3 epochs)
- Cost: ¥25,920-34,560 (~$3,600-4,800)
- Model quality: Maximum (full dataset, full training)

---

### Strategy 4: Hybrid Approach (Best Value)

**Best for**: Balancing cost, time, and quality

**Approach**: Combine sampling + optimization

1. **Sample 10M diverse sentences** (not random - use diversity sampling)
2. **Train for 2 epochs** on A100 GPU
3. **Use mixed precision** (FP16) for 2x speedup
4. **Optimize batch size** to maximize GPU utilization

**Configuration**:
```yaml
data:
  sample_size: 10000000  # 10M sentences
  sample_strategy: "diverse"  # Maximize vocabulary coverage

teacher:
  num_epochs: 2
  batch_size: 24  # Larger on A100
  gradient_accumulation_steps: 2
  fp16: true  # Mixed precision
  max_seq_length: 384  # Reduce if most sentences are shorter
```

**Expected results**:
- Training time: 4-6 days total
- Cost: ¥4,800-7,200 (~$670-1,000)
- Model quality: 96-98% of full training
- Best cost-to-quality ratio

---

## Optimizations for Large-Scale Training

### 1. Data Loading Optimization

**Problem**: Loading 71M sentences into memory = out of memory

**Solution**: Streaming data loader

```python
# Add to data_processor.py
class StreamingDataset(paddle.io.IterableDataset):
    """Load data on-the-fly without keeping in memory"""
    
    def __init__(self, data_path, batch_size=1000):
        self.data_path = data_path
        self.batch_size = batch_size
    
    def __iter__(self):
        # Yield batches from disk, never load all at once
        with open(self.data_path, 'rb') as f:
            while True:
                batch = self.load_batch(f, self.batch_size)
                if not batch:
                    break
                for item in batch:
                    yield item
```

### 2. Gradient Checkpointing

**Problem**: 7B model = huge memory footprint

**Solution**: Trade compute for memory

```python
# In model initialization
model.gradient_checkpointing_enable()

# Saves 40-50% GPU memory
# Allows 2x larger batch sizes
# Only 10-15% slower
```

### 3. Optimized Tokenization

**Problem**: Tokenizing 71M sentences is slow

**Solution**: Pre-tokenize and cache

```bash
# Do this once, save results
python -c "
from src.data_processor import DataProcessor
processor = DataProcessor(config)
processor.tokenize_and_save()  # Save tokenized data to disk
"

# Training loads pre-tokenized data
# 10x faster startup
```

### 4. Mixed Precision Training (FP16)

**Already in config**, but ensure it's enabled:

```yaml
teacher:
  fp16: true  # Use 16-bit floats instead of 32-bit
  
  # Optional: gradient scaling to prevent underflow
  fp16_opt_level: "O2"  # O1=conservative, O2=aggressive
```

**Benefits**:
- 2x faster training
- 50% less memory
- Minimal quality loss (<0.5% BLEU)

### 5. Dynamic Padding

**Problem**: Padding all sequences to 512 tokens wastes computation

**Solution**: Pad to longest in batch

```python
def smart_collate(batch):
    # Find max length in THIS batch only
    max_len = max(len(x['input_ids']) for x in batch)
    
    # Pad only to max_len, not global max (512)
    # Saves 30-40% compute on average
```

### 6. Learning Rate Scheduling

For large datasets, optimize LR schedule:

```yaml
teacher:
  learning_rate: 5.0e-5
  warmup_steps: 2000  # Longer warmup for stability
  lr_scheduler: "cosine"  # Cosine annealing
  min_lr: 1.0e-6  # Don't decay to zero
```

---

## Updated Configuration for 71M Dataset

Here's the optimized `config.yaml` for your use case:

```yaml
# Recommended config for 71M sentence dataset

data:
  # Option 1: Sample (Recommended)
  sample_size: 10000000  # 10M high-quality sentences
  sample_strategy: "diverse"
  quality_threshold: 0.75
  
  # Option 2: Full dataset
  # sample_size: null  # Use all 71M
  
  chunk_size: 50000  # Larger chunks for efficiency

teacher:
  model_name: "Qwen/Qwen2.5-7B"
  use_paddlenlp: true
  
  output_dir: "models/teacher"
  num_epochs: 2  # Reduced from 3
  batch_size: 24  # Optimized for A100
  learning_rate: 5.0e-5
  warmup_steps: 2000  # Longer warmup
  max_seq_length: 384  # Reduced if appropriate
  gradient_accumulation_steps: 2
  
  # Optimizations
  fp16: true
  gradient_checkpointing: true
  dataloader_num_workers: 4
  
  # Large dataset settings
  save_steps: 10000
  eval_steps: 5000
  logging_steps: 100
  save_total_limit: 3  # Keep only 3 checkpoints
  
  # Optional: Early stopping
  early_stopping_patience: 3
  early_stopping_threshold: 0.001

student:
  model_name: "Qwen/Qwen2.5-0.5B"
  num_epochs: 3  # Can reduce to 2 if time-constrained
  batch_size: 32  # Smaller model = larger batch
  learning_rate: 3.0e-4
  
  # Distillation from teacher
  distillation_alpha: 0.7  # 70% teacher, 30% labels
  temperature: 3.0
```

---

## Monitoring Long-Running Jobs

### 1. Setup CloudMonitor Alerts

In Alibaba Cloud console:
```
CloudMonitor > Alarm Rules > Create:
- GPU utilization < 80% for 30 min (underutilized)
- Instance stops unexpectedly (training crashed)
- Disk usage > 90% (running out of space)
```

### 2. Training Progress Dashboard

```bash
# Install TensorBoard
pip install tensorboard

# In training script, add logging:
from paddle.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='logs/tensorboard')

# Log metrics every step:
writer.add_scalar('train/loss', loss, step)
writer.add_scalar('train/learning_rate', lr, step)

# View progress:
tensorboard --logdir logs/tensorboard --host 0.0.0.0 --port 6006

# Access from browser:
# http://[YOUR_IP]:6006
```

### 3. Remote Monitoring Script

Run this locally to check training progress:

```powershell
# monitor_training.ps1
param([string]$IP, [string]$LogFile = "/opt/linguabridge/logs/training.log")

while ($true) {
    $lastLines = ssh root@$IP "tail -50 $LogFile"
    Clear-Host
    Write-Host "=== Training Monitor ===" -ForegroundColor Cyan
    Write-Host "Server: $IP" -ForegroundColor Gray
    Write-Host "Time: $(Get-Date)" -ForegroundColor Gray
    Write-Host ""
    Write-Host $lastLines
    Start-Sleep -Seconds 30
}
```

### 4. Telegram/Email Alerts

```python
# Add to training script
def send_alert(message):
    # Use Telegram Bot API or email
    import requests
    bot_token = "YOUR_TELEGRAM_BOT_TOKEN"
    chat_id = "YOUR_CHAT_ID"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": message})

# Send alerts at key events:
send_alert("✅ Training started")
send_alert(f"📊 Epoch 1 complete, loss: {loss:.4f}")
send_alert(f"⚠️ Validation loss increasing (possible overfit)")
send_alert("🎉 Training complete!")
```

---

## Cost Optimization for Long Training

### 1. Use Preemptible Instances (Spot Instances)

Save 80-90% on GPU costs, but can be interrupted:

```bash
# Create preemptible instance via CLI
aliyun ecs CreateInstance \
  --InstanceType ecs.gn7i-c16g1.4xlarge \
  --SpotStrategy SpotAsPriceGo \
  --SpotDuration 6  # Hours

# Implement checkpoint resuming in training script
# If interrupted, restart from last checkpoint
```

**Best for**: Non-urgent training with good checkpointing

### 2. Schedule Training for Off-Peak Hours

Some regions have time-based pricing:

```bash
# Start training at 00:00, stop at 08:00
# Resume at 18:00, run overnight
# Can save 20-30% in some regions
```

### 3. Reserved Instances (Long-term)

If training will take 30+ days:

```bash
# Buy 1-month reserved instance (30-40% discount)
# Via Console: ECS > Reserved Instances
# Apply to running instances automatically
```

### 4. Cross-Region Price Comparison

| Region | A10 Instance Cost | Savings |
|--------|------------------|---------|
| cn-beijing | ¥30/hour | Baseline |
| cn-shanghai | ¥32/hour | -7% |
| cn-shenzhen | ¥28/hour | +7% |
| cn-hongkong | ¥35/hour | -17% |

Check current prices: https://www.aliyun.com/price

---

## Troubleshooting Long Training Runs

### Issue: Training Slower Than Expected

**Diagnosis**:
```bash
# Check GPU utilization
nvidia-smi -l 1

# Should see:
# - GPU Utilization: 95-100%
# - Memory Usage: 20-23GB / 24GB
# - Power: 250-300W / 300W
```

**Fixes**:
- Increase batch size (if GPU not fully utilized)
- Enable mixed precision (FP16)
- Check dataloader (`num_workers=4`)
- Verify no CPU bottleneck (`htop`)

### Issue: Out of Memory

**Solutions**:
```python
# 1. Enable gradient checkpointing
model.gradient_checkpointing_enable()

# 2. Reduce batch size
batch_size = 4  # Down from 8

# 3. Reduce sequence length
max_seq_length = 256  # Down from 512

# 4. Use gradient accumulation
gradient_accumulation_steps = 8  # Effective batch: 4*8=32
```

### Issue: Loss Not Decreasing

**After 100k steps**, if loss plateaus:

```yaml
# Increase learning rate
learning_rate: 1.0e-4  # From 5e-5

# Or restart with different warmup
warmup_steps: 5000  # From 2000

# Or try different optimizer
optimizer: "adamw"  # vs "adam"
```

### Issue: Training Diverges (Loss → NaN)

**Immediate fixes**:
```yaml
# 1. Lower learning rate
learning_rate: 1.0e-5  # Much lower

# 2. Increase warmup
warmup_steps: 5000

# 3. Disable FP16 temporarily
fp16: false

# 4. Add gradient clipping
max_grad_norm: 0.5  # From 1.0
```

---

## Final Recommendations for 71M Dataset

### For Budget-Conscious Projects:
✅ **Sample 5-10M sentences** (quality or diverse strategy)
✅ **Train on A10 GPU** (good performance/cost ratio)
✅ **2 epochs maximum**
✅ **Use FP16 mixed precision**

**Result**: 5-7 days, ¥3,600-5,000 (~$500-700), excellent quality

### For Time-Sensitive Projects:
✅ **Sample 10M sentences** (diverse strategy)
✅ **Train on A100 GPU** (fastest single GPU)
✅ **2 epochs with aggressive optimization**
✅ **Enable all speedups** (FP16, checkpointing, etc.)

**Result**: 3-4 days, ¥3,600-4,800 (~$500-670), excellent quality

### For Maximum Quality (No Budget Limit):
✅ **Use full 71M dataset**
✅ **Multi-GPU training** (4× A10 or A100)
✅ **3 epochs with careful tuning**
✅ **Extensive validation and early stopping**

**Result**: 7-10 days, ¥25,000-36,000 (~$3,500-5,000), maximum quality

---

## Conclusion

**For most use cases, I strongly recommend Strategy 4 (Hybrid)**:
- 10M diverse samples
- A100 GPU
- 2 epochs
- All optimizations enabled
- **4-6 days, ¥5,000-7,000 (~$700-1,000)**

This gives you 96-98% of the quality at 13-17% of the cost and time.

Remember: Modern LLMs are data-efficient. **Quality > Quantity** for translation tasks.

---

*Last updated: January 12, 2026*
