# Alibaba Cloud Deployment Guide

This guide covers deploying LinguaBridge Local on Alibaba Cloud for production use.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Cloud Resources Required](#cloud-resources-required)
3. [Deployment Strategy](#deployment-strategy)
4. [Step-by-Step Setup](#step-by-step-setup)
5. [Cost Optimization](#cost-optimization)
6. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Architecture Overview

### Recommended Alibaba Cloud Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Alibaba Cloud Setup                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 1: Teacher Training (GPU Instance)                   │
│  ├─ ECS gn7i Instance (NVIDIA A10)                          │
│  ├─ 16 vCPU, 60GB RAM                                       │
│  ├─ Train Qwen2.5-7B teacher model                          │
│  └─ Generate soft labels (then terminate)                   │
│                                                              │
│  Phase 2: Student Distillation (CPU Instance)               │
│  ├─ ECS c7 Instance (ARM or x86)                            │
│  ├─ 8 vCPU, 16GB RAM                                        │
│  ├─ Distill Qwen2.5-0.5B student model                      │
│  └─ Save to OSS for deployment                              │
│                                                              │
│  Phase 3: Production Deployment                             │
│  ┌──────────────────────────────────────────────┐          │
│  │  SLB (Server Load Balancer)                  │          │
│  │  Public IP with SSL                          │          │
│  └──────────────────┬───────────────────────────┘          │
│                     │                                        │
│        ┌────────────┼────────────┐                          │
│        │            │            │                           │
│  ┌─────▼─────┐ ┌───▼──────┐ ┌──▼───────┐                  │
│  │ ECS c7    │ │ ECS c7   │ │ ECS c7   │                  │
│  │ API       │ │ API      │ │ API      │                  │
│  │ Service   │ │ Service  │ │ Service  │                  │
│  │ (FastAPI) │ │ (FastAPI)│ │ (FastAPI)│                  │
│  └─────┬─────┘ └────┬─────┘ └────┬─────┘                  │
│        │            │            │                           │
│        └────────────┼────────────┘                          │
│                     │                                        │
│                ┌────▼─────┐                                 │
│                │   OSS    │                                 │
│                │  Models  │                                 │
│                │  & Cache │                                 │
│                └──────────┘                                 │
│                                                              │
│  Monitoring: CloudMonitor + Log Service                     │
│  Security: Security Group + WAF                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Cloud Resources Required

### 1. ECS Instances (Elastic Compute Service)

#### For Teacher Training (Temporary)
- **Instance Type**: ecs.gn7i-c8g1.2xlarge
- **Specs**: 8 vCPU, 30GB RAM, 1x NVIDIA A10 (24GB)
- **OS**: Ubuntu 22.04 LTS
- **Storage**: 200GB ESSD PL1
- **Usage**: Train teacher model (1-3 days), then terminate
- **Cost**: ~¥15/hour (pay-as-you-go)

#### For Student Distillation (Temporary)
- **Instance Type**: ecs.c7.2xlarge
- **Specs**: 8 vCPU, 16GB RAM
- **OS**: Ubuntu 22.04 LTS
- **Storage**: 100GB ESSD PL0
- **Usage**: Distill student model (1-2 days), then terminate
- **Cost**: ~¥1.2/hour

#### For Production API (Persistent)
- **Instance Type**: ecs.c7.xlarge (or ARM: ecs.c7a.xlarge)
- **Specs**: 4 vCPU, 8GB RAM
- **OS**: Ubuntu 22.04 LTS
- **Storage**: 50GB ESSD PL0
- **Quantity**: 2-3 instances (for HA)
- **Cost**: ~¥0.6/hour per instance

### 2. OSS (Object Storage Service)
- **Bucket**: Private bucket for models and data
- **Storage**: ~5GB (models + processed data)
- **Traffic**: Minimal (internal VPC access)
- **Cost**: ~¥0.12/GB/month + ¥0.5/GB transfer

### 3. SLB (Server Load Balancer)
- **Type**: Application Load Balancer (Layer 7)
- **Bandwidth**: 10Mbps (elastic)
- **SSL**: Free Alibaba Cloud certificate
- **Cost**: ~¥0.3/hour + traffic

### 4. VPC (Virtual Private Cloud)
- **Network**: Free
- **NAT Gateway**: For internet access (optional)
- **Cost**: ¥0.5/hour if needed

### 5. CloudMonitor & Log Service
- **CloudMonitor**: Free (basic metrics)
- **Log Service**: ¥0.5/GB (optional detailed logs)

### 6. Domain & DNS
- **Domain**: Register via Alibaba Cloud
- **DNS**: Free with domain

---

## Deployment Strategy

### Option 1: Full Cloud Training + Deployment
**Best for**: Complete production setup with model training

**Steps**:
1. Train teacher model on GPU instance (¥15/hr × 24-72 hrs)
2. Distill student model on CPU instance (¥1.2/hr × 12-48 hrs)
3. Deploy API services with load balancer
4. Total time: 4-7 days
5. Total cost: ~¥600-2000 (training) + ¥50/month (production)

### Option 2: Upload Pre-trained Models
**Best for**: Quick deployment, models already trained locally

**Steps**:
1. Train models locally (or use existing)
2. Upload to OSS via ossutil
3. Deploy API services immediately
4. Total time: 2-4 hours
5. Total cost: ~¥50/month (production only)

### Option 3: Hybrid (Recommended)
**Best for**: Cost optimization

**Steps**:
1. Do data processing locally (free)
2. Train teacher on GPU instance (¥15/hr × 24-72 hrs)
3. Distill student locally on your Surface Pro (free)
4. Upload final model to OSS
5. Deploy API services
6. Total cost: ~¥400-1200 (training) + ¥50/month (production)

---

## Step-by-Step Setup

### Phase 1: Alibaba Cloud Account Setup

#### 1.1 Create Account & Setup
```bash
# Register at: https://www.alibabacloud.com
# Complete real-name verification (required in China)
# Add payment method (支付宝/WeChat Pay/Credit Card)
```

#### 1.2 Install Alibaba Cloud CLI
```powershell
# Download from: https://www.alibabacloud.com/help/cli
# Or use pip (if available)
pip install aliyun-cli

# Configure credentials
aliyun configure
# Enter Access Key ID and Secret
```

#### 1.3 Install ossutil (for file uploads)
```powershell
# Download: https://www.alibabacloud.com/help/oss/developer-reference/ossutil
# Windows: Download ossutil64.exe
# Rename to ossutil.exe and add to PATH

# Configure
ossutil config
# Enter endpoint (e.g., oss-cn-beijing.aliyuncs.com)
# Enter Access Key ID and Secret
```

---

### Phase 2: Setup Cloud Infrastructure

#### 2.1 Create VPC and Security Groups

**Via Web Console**:
1. Navigate to VPC Console
2. Create VPC: `linguabridge-vpc`
   - CIDR: 172.16.0.0/12
   - Region: cn-beijing (or nearest)
3. Create vSwitch: `linguabridge-subnet`
   - CIDR: 172.16.0.0/24
   - Zone: cn-beijing-h

4. Create Security Group: `linguabridge-sg`
   - Allow inbound:
     - SSH (22) from your IP
     - HTTP (80) from 0.0.0.0/0
     - HTTPS (443) from 0.0.0.0/0
     - API port (8000) from SLB only

**Via CLI**:
```bash
# Create VPC
aliyun vpc CreateVpc --CidrBlock 172.16.0.0/12 --VpcName linguabridge-vpc

# Create Security Group (simplified)
aliyun ecs CreateSecurityGroup --SecurityGroupName linguabridge-sg \
  --VpcId vpc-xxxxxxx
```

#### 2.2 Create OSS Bucket

**Via Web Console**:
1. Navigate to OSS Console
2. Create Bucket
   - Name: `linguabridge-models-[your-unique-id]`
   - Region: Same as ECS (cn-beijing)
   - ACL: Private
   - Versioning: Enabled
   - Storage Class: Standard

**Via CLI**:
```bash
ossutil mb oss://linguabridge-models-[your-unique-id]
```

---

### Phase 3: Train Models (Cloud or Local)

#### Option A: Train on Alibaba Cloud GPU Instance

**3.1 Launch GPU Instance**
```bash
# Via Web Console (easier):
# 1. Navigate to ECS Console
# 2. Click "Create Instance"
# 3. Select:
#    - Instance Type: ecs.gn7i-c8g1.2xlarge (GPU)
#    - Image: Ubuntu 22.04 LTS
#    - System Disk: 200GB ESSD PL1
#    - VPC: linguabridge-vpc
#    - Security Group: linguabridge-sg
#    - Login: Password or Key Pair
# 4. Launch and wait for "Running" status
```

**3.2 Connect to GPU Instance**
```bash
# Get public IP from console
ssh root@[GPU_INSTANCE_IP]
```

**3.3 Setup Environment**
```bash
# Update system
apt update && apt upgrade -y

# Install Python and dependencies
apt install -y python3.10 python3.10-venv python3-pip git

# Install NVIDIA drivers (usually pre-installed)
nvidia-smi  # Verify GPU

# Create project directory
mkdir -p /opt/linguabridge
cd /opt/linguabridge

# Upload your code
# Option 1: Git (if you have a repo)
git clone https://github.com/yourusername/CH-EN-LLM.git .

# Option 2: Upload via SCP from local
# On your Windows machine:
# scp -r C:\Users\noahg\Documents\CH-EN-LLM root@[GPU_INSTANCE_IP]:/opt/linguabridge/

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install paddlepaddle-gpu==3.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
pip install -r requirements.txt
```

**3.4 Upload Training Data**
```bash
# On your Windows machine, upload data to OSS
ossutil cp data/raw/en.txt oss://linguabridge-models-[your-id]/data/raw/
ossutil cp data/raw/zh.txt oss://linguabridge-models-[your-id]/data/raw/

# On GPU instance, download from OSS
ossutil cp oss://linguabridge-models-[your-id]/data/raw/en.txt data/raw/
ossutil cp oss://linguabridge-models-[your-id]/data/raw/zh.txt data/raw/
```

**3.5 Run Training Pipeline**
```bash
# Process data
python run.py process

# Train teacher model
python run.py train

# Upload teacher model to OSS
ossutil cp -r models/teacher/ oss://linguabridge-models-[your-id]/models/teacher/
```

**3.6 Generate Soft Labels (for distillation)**
```bash
# Run soft label generation (if implemented)
# This creates teacher predictions for student training
python -c "from src.train_teacher import generate_soft_labels; generate_soft_labels()"

# Upload soft labels
ossutil cp -r data/soft_labels/ oss://linguabridge-models-[your-id]/data/soft_labels/
```

**3.7 Terminate GPU Instance** (Save costs!)
```bash
# Via Console: Stop or Release the instance
# Or via CLI:
aliyun ecs StopInstance --InstanceId i-xxxxxxx
```

#### Option B: Train Locally & Upload

**3.1 Train on Your Surface Pro**
```powershell
# In your local project directory
cd C:\Users\noahg\Documents\CH-EN-LLM

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run training pipeline
python run.py process
python run.py train     # Teacher (slow on CPU)
python run.py distill   # Student
```

**3.2 Upload to OSS**
```powershell
# Upload models
ossutil cp -r models\teacher\ oss://linguabridge-models-[your-id]/models/teacher/ --recursive
ossutil cp -r models\student\ oss://linguabridge-models-[your-id]/models/student/ --recursive

# Upload processed data
ossutil cp -r data\processed\ oss://linguabridge-models-[your-id]/data/processed/ --recursive
```

---

### Phase 4: Deploy Production API

#### 4.1 Launch Production ECS Instances

**Create API Instance (repeat for 2-3 instances)**
```bash
# Via Console:
# 1. ECS Console > Create Instance
# 2. Select:
#    - Type: ecs.c7.xlarge (4 vCPU, 8GB RAM)
#    - Image: Ubuntu 22.04 LTS
#    - Disk: 50GB ESSD PL0
#    - VPC: linguabridge-vpc
#    - Security Group: linguabridge-sg
#    - Name: linguabridge-api-01
# 3. Launch

# Get instance IPs and save them
```

#### 4.2 Setup Each API Instance

**Connect and Setup**
```bash
# SSH to instance
ssh root@[API_INSTANCE_IP]

# Install dependencies
apt update && apt install -y python3.10 python3.10-venv python3-pip git

# Install ossutil
wget http://gosspublic.alicdn.com/ossutil/1.7.18/ossutil64
chmod 755 ossutil64
mv ossutil64 /usr/local/bin/ossutil
ossutil config  # Configure with your credentials

# Create project directory
mkdir -p /opt/linguabridge
cd /opt/linguabridge

# Download code (via git or upload)
git clone https://github.com/yourusername/CH-EN-LLM.git .
# OR upload via scp

# Setup virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download models from OSS
mkdir -p models/student
ossutil cp -r oss://linguabridge-models-[your-id]/models/student/ models/student/ --recursive

# Download processed data (if needed)
mkdir -p data/processed
ossutil cp -r oss://linguabridge-models-[your-id]/data/processed/ data/processed/ --recursive
```

#### 4.3 Create Systemd Service

**Create service file**
```bash
cat > /etc/systemd/system/linguabridge-api.service << 'EOF'
[Unit]
Description=LinguaBridge Translation API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/linguabridge
Environment="PATH=/opt/linguabridge/.venv/bin"
ExecStart=/opt/linguabridge/.venv/bin/python -m uvicorn src.app_api:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
systemctl daemon-reload
systemctl enable linguabridge-api
systemctl start linguabridge-api
systemctl status linguabridge-api

# Check logs
journalctl -u linguabridge-api -f
```

#### 4.4 Test API
```bash
# Test locally on instance
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you?", "source": "en", "target": "zh"}'

# Should return Chinese translation
```

---

### Phase 5: Setup Load Balancer

#### 5.1 Create SLB Instance

**Via Web Console**:
1. Navigate to SLB Console
2. Create Instance
   - Name: linguabridge-slb
   - Type: Application Load Balancer
   - Network: VPC (linguabridge-vpc)
   - IP Version: IPv4
   - Billing: Pay-As-You-Go
3. Configure Listener
   - Protocol: HTTP (port 80)
   - Backend: Create new backend server group
   - Add your API instances (port 8000)
   - Health Check: 
     - Path: /health
     - Interval: 10s
     - Timeout: 5s

4. (Optional) Add HTTPS Listener
   - Protocol: HTTPS (port 443)
   - SSL Certificate: Apply free certificate
   - Backend: Same server group

#### 5.2 Configure Backend Server Group
```bash
# Via Console:
# 1. Add all API instances (linguabridge-api-01, -02, -03)
# 2. Backend port: 8000
# 3. Weight: 100 for each
# 4. Health check: Enabled
```

#### 5.3 Test Load Balancer
```bash
# Get SLB public IP from console
curl -X POST http://[SLB_PUBLIC_IP]/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "source": "en", "target": "zh"}'
```

---

### Phase 6: Domain and SSL

#### 6.1 Setup Domain (Optional)
```bash
# Register domain via Alibaba Cloud (if needed)
# Or use existing domain

# Add DNS record:
# Type: A
# Host: api
# Value: [SLB_PUBLIC_IP]
# Result: api.yourdomain.com -> SLB
```

#### 6.2 Configure SSL Certificate
```bash
# Via Certificate Service:
# 1. Apply for free DV SSL (20 free per year)
# 2. Domain validation
# 3. Download certificate
# 4. Upload to SLB HTTPS listener
```

---

## Cost Optimization

### Training Phase (One-time)

| Resource | Specs | Duration | Cost |
|----------|-------|----------|------|
| GPU Instance (Teacher) | ecs.gn7i (A10) | 48 hours | ¥720 |
| CPU Instance (Distill) | ecs.c7.2xlarge | 24 hours | ¥29 |
| OSS Storage | 5GB | 1 month | ¥0.6 |
| **Total** | | | **¥750** |

### Production Phase (Monthly)

| Resource | Specs | Quantity | Cost/Month |
|----------|-------|----------|------------|
| API Instances | ecs.c7.xlarge | 2 | ¥864 |
| SLB | 10Mbps | 1 | ¥216 |
| OSS Storage | 5GB | 1 | ¥0.6 |
| OSS Traffic | 10GB/month | - | ¥5 |
| CloudMonitor | Basic | - | Free |
| **Total** | | | **¥1,086/month** |

### Cost Saving Tips

1. **Use Reserved Instances**: Save 30-60% for 1-year commitment
2. **Use Preemptible Instances**: Save 80% for GPU training (if flexible)
3. **Auto Scaling**: Scale down during low traffic (nights/weekends)
4. **OSS Lifecycle**: Archive old models to IA storage
5. **Regional Selection**: Some regions are cheaper (cn-beijing vs cn-hangzhou)
6. **Bandwidth Optimization**: Use internal VPC for inter-service communication

---

## Monitoring & Maintenance

### Setup CloudMonitor Alerts

**Via Console**:
1. CloudMonitor > Alarm Rules
2. Create Alert Rules:
   - CPU > 80% for 5 minutes
   - Memory > 90% for 5 minutes
   - Disk > 85%
   - API instance down
   - SLB unhealthy targets > 0

### Setup Log Service (Optional)

**Collect API Logs**:
```bash
# On each API instance
# Install Log Agent
wget http://logtail-release-cn-beijing.oss-cn-beijing-internal.aliyuncs.com/linux64/logtail.sh
sh logtail.sh install cn-beijing

# Configure log collection for:
# - /opt/linguabridge/logs/api.log
# - systemd logs (journalctl)
```

### Regular Maintenance Tasks

**Weekly**:
- Check CloudMonitor dashboards
- Review error logs
- Test translation quality
- Check disk usage

**Monthly**:
- Review costs and optimize
- Update dependencies (security patches)
- Backup models to OSS Archive
- Performance tuning

**Quarterly**:
- Retrain models with new data
- Upgrade to newer Qwen versions
- Scale up/down based on usage

---

## Deployment Checklist

### Pre-Deployment
- [ ] Alibaba Cloud account created and verified
- [ ] Payment method added
- [ ] Access keys generated
- [ ] CLI tools installed (aliyun, ossutil)
- [ ] Models trained (locally or cloud)

### Infrastructure
- [ ] VPC and vSwitch created
- [ ] Security groups configured
- [ ] OSS bucket created
- [ ] Models uploaded to OSS

### API Deployment
- [ ] ECS instances launched (2-3)
- [ ] All instances configured with code
- [ ] Models downloaded from OSS
- [ ] Systemd services created and running
- [ ] Local API tests passed

### Load Balancing
- [ ] SLB instance created
- [ ] Backend server group configured
- [ ] Health checks enabled
- [ ] Public IP assigned
- [ ] External API tests passed

### Optional
- [ ] Domain configured
- [ ] SSL certificate applied
- [ ] CloudMonitor alerts set
- [ ] Log Service configured
- [ ] Auto-scaling configured

### Production Ready
- [ ] Load testing completed
- [ ] Monitoring dashboards set up
- [ ] Backup procedures documented
- [ ] Incident response plan ready

---

## Troubleshooting

### Common Issues

**Issue 1: Cannot connect to ECS instance**
```bash
# Check:
# 1. Security group allows SSH (port 22) from your IP
# 2. Instance has public IP assigned
# 3. Key pair or password correct
# 4. Use correct username (root or ubuntu)
```

**Issue 2: ossutil upload fails**
```bash
# Check:
# 1. Credentials configured correctly (ossutil config)
# 2. Bucket exists and accessible
# 3. Network connection stable
# 4. Endpoint matches bucket region

# Re-configure:
ossutil config
```

**Issue 3: API service won't start**
```bash
# Check logs
journalctl -u linguabridge-api -n 50

# Common fixes:
# - Missing dependencies: pip install -r requirements.txt
# - Port already in use: netstat -tlnp | grep 8000
# - Model files missing: check models/student/
# - Python version: python3.10 --version
```

**Issue 4: SLB health check failing**
```bash
# On API instance:
# 1. Check service running: systemctl status linguabridge-api
# 2. Test health endpoint: curl http://localhost:8000/health
# 3. Check firewall: ufw status (should be inactive)
# 4. Check logs: journalctl -u linguabridge-api -f

# Add health check endpoint to app_api.py if missing:
# @app.get("/health")
# def health():
#     return {"status": "healthy"}
```

**Issue 5: Out of memory errors**
```bash
# Monitor memory
free -h
top

# Solutions:
# 1. Increase instance RAM (resize instance)
# 2. Reduce model batch size in config.yaml
# 3. Add swap space (temporary):
fallocate -l 4G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
```

---

## Next Steps

After successful deployment:

1. **Performance Tuning**
   - Optimize batch size and worker count
   - Enable model quantization (INT8)
   - Add Redis for caching frequent translations

2. **Feature Enhancements**
   - Add user authentication
   - Implement rate limiting
   - Add translation memory
   - Support batch translations

3. **Scaling**
   - Setup auto-scaling groups
   - Add more regions (multi-region deployment)
   - Implement CDN for global access

4. **Documentation**
   - Create API documentation (Swagger/OpenAPI)
   - Write user guides
   - Create admin runbooks

---

## Support Resources

- **Alibaba Cloud Docs**: https://www.alibabacloud.com/help
- **ECS User Guide**: https://www.alibabacloud.com/help/ecs
- **OSS User Guide**: https://www.alibabacloud.com/help/oss
- **SLB User Guide**: https://www.alibabacloud.com/help/slb
- **Community**: https://www.alibabacloud.com/forum

---

## Summary

This guide covered deploying LinguaBridge on Alibaba Cloud with:
- ✅ GPU training for teacher model
- ✅ CPU distillation for student model
- ✅ Production API deployment with high availability
- ✅ Load balancing and auto-scaling
- ✅ Monitoring and alerting
- ✅ Cost optimization strategies

**Estimated Setup Time**: 1-2 days
**Estimated Monthly Cost**: ¥1,000-1,500 (~$140-210 USD)

Good luck with your deployment! 🚀
