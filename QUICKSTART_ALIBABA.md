# Quick Start - Alibaba Cloud Deployment

This is a streamlined guide to get your LinguaBridge API running on Alibaba Cloud in 30 minutes.

## Prerequisites

- [ ] Alibaba Cloud account with payment method
- [ ] Access Key ID and Secret (from RAM console)
- [ ] Models trained locally (student model ready)
- [ ] This project code

## Option 1: Quick Deploy (Use Pre-trained Models)

### Step 1: Setup Local Tools (5 minutes)

**Install ossutil on Windows:**
```powershell
# Download from: https://www.alibabacloud.com/help/oss/developer-reference/ossutil
# Or direct link:
# https://gosspublic.alicdn.com/ossutil/ossutilwindows64.zip

# Extract and add to PATH, then configure:
ossutil config
# Enter:
# - Endpoint: oss-cn-beijing.aliyuncs.com (or your region)
# - AccessKeyID: your_key_id
# - AccessKeySecret: your_secret
```

### Step 2: Create OSS Bucket (3 minutes)

**Via Web Console:**
1. Go to https://oss.console.aliyun.com
2. Click "Create Bucket"
3. Bucket Name: `linguabridge-models-[random]` (must be unique)
4. Region: cn-beijing (Beijing)
5. ACL: Private
6. Click Create

**Save your bucket name!** You'll need it for all commands below.

### Step 3: Upload Models (5 minutes)

```powershell
# Navigate to your project
cd C:\Users\noahg\Documents\CH-EN-LLM

# Upload student model (required)
ossutil cp -r models\student\ oss://your-bucket-name/models/student/ --update

# Upload processed data (vocabularies, required)
ossutil cp -r data\processed\ oss://your-bucket-name/data/processed/ --update

# Upload config
ossutil cp config.yaml oss://your-bucket-name/config.yaml

# Verify upload
ossutil ls oss://your-bucket-name/ -r
```

Or use the automated script:
```powershell
.\scripts\upload_to_oss.ps1 -BucketName "your-bucket-name"
```

### Step 4: Create ECS Instance (5 minutes)

**Via Web Console:**
1. Go to https://ecs.console.aliyun.com
2. Click "Create Instance"
3. **Billing**: Pay-As-You-Go
4. **Region**: cn-beijing (same as OSS)
5. **Instance Type**: 
   - Category: Compute Optimized
   - Type: ecs.c7.xlarge (4 vCPU, 8GB RAM)
6. **Image**: Ubuntu 22.04 64-bit
7. **Storage**: 50GB ESSD PL0 (system disk)
8. **Network**: 
   - VPC: Create new or use default
   - Public IP: Assign (5Mbps)
   - Security Group: Create new
     - Allow: SSH (22), HTTP (80), Custom TCP (8000)
9. **Login**: 
   - Set root password (remember it!)
10. Click **Create**

Wait 2-3 minutes for instance to start, then note the **Public IP**.

### Step 5: Setup ECS Instance (10 minutes)

**SSH to your instance:**
```bash
ssh root@YOUR_PUBLIC_IP
# Enter password when prompted
```

**Upload and run setup script:**
```bash
# Download setup script from your local machine
# On Windows, use SCP:
# scp C:\Users\noahg\Documents\CH-EN-LLM\scripts\deploy_setup_alibaba.sh root@YOUR_PUBLIC_IP:/root/

# Or create it directly on server:
wget https://raw.githubusercontent.com/YOUR_REPO/main/scripts/deploy_setup_alibaba.sh
# OR paste the script content manually

# Make executable
chmod +x deploy_setup_alibaba.sh

# Run setup (provide your Git repo URL if available, or leave empty)
bash deploy_setup_alibaba.sh [your-git-repo-url]

# Configure ossutil on server
ossutil config
# Use same credentials as local

# Download models from OSS
cd /opt/linguabridge
ossutil cp -r oss://your-bucket-name/models/student/ models/student/
ossutil cp -r oss://your-bucket-name/data/processed/ data/processed/
ossutil cp oss://your-bucket-name/config.yaml config.yaml
```

**Upload your code (if not using Git):**
```bash
# On your Windows machine:
cd C:\Users\noahg\Documents\CH-EN-LLM

# Upload entire project (from local PowerShell)
scp -r src\ root@YOUR_PUBLIC_IP:/opt/linguabridge/
scp -r requirements.txt config.yaml run.py root@YOUR_PUBLIC_IP:/opt/linguabridge/
```

### Step 6: Start API Service (2 minutes)

**On ECS instance:**
```bash
# Activate virtual environment
cd /opt/linguabridge
source .venv/bin/activate

# Test if models are present
ls -lh models/student/

# Start service
systemctl start linguabridge-api

# Check status
systemctl status linguabridge-api

# View logs
journalctl -u linguabridge-api -f
# Press Ctrl+C to exit logs

# Test locally
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "source": "en", "target": "zh"}'

# Should return Chinese translation
```

**Enable on boot:**
```bash
systemctl enable linguabridge-api
```

### Step 7: Test from Your Computer (2 minutes)

```powershell
# On your Windows machine
# Test health check
curl http://YOUR_PUBLIC_IP:8000/health

# Test translation
curl -X POST http://YOUR_PUBLIC_IP:8000/translate `
  -H "Content-Type: application/json" `
  -d '{\"text\": \"Good morning\", \"source\": \"en\", \"target\": \"zh\"}'

# Or use test script
.\scripts\test_api.ps1 -ApiUrl "http://YOUR_PUBLIC_IP:8000"
```

### Done! 🎉

Your API is now running at: `http://YOUR_PUBLIC_IP:8000`

**API Documentation**: Visit `http://YOUR_PUBLIC_IP:8000/docs` in your browser

---

## Option 2: Production Setup (with Load Balancer)

Follow the full guide in [ALIBABA_CLOUD_DEPLOYMENT.md](ALIBABA_CLOUD_DEPLOYMENT.md) for:
- High availability (multiple instances)
- Load balancer with SSL
- Auto-scaling
- Monitoring and alerts
- Domain setup

---

## Common Issues

### Issue: Cannot connect to instance
```bash
# Check:
# 1. Security group allows port 8000 from 0.0.0.0/0
# 2. Service is running: systemctl status linguabridge-api
# 3. Port is listening: netstat -tlnp | grep 8000
```

### Issue: Model not found
```bash
# Download again from OSS
cd /opt/linguabridge
ossutil cp -r oss://your-bucket-name/models/student/ models/student/ --update
```

### Issue: Service fails to start
```bash
# Check logs
journalctl -u linguabridge-api -n 100

# Common fixes:
# - Missing dependencies: pip install -r requirements.txt
# - Wrong Python version: python3.10 --version
# - Config file missing: check config.yaml exists
# - Port in use: lsof -i :8000

# Restart service
systemctl restart linguabridge-api
```

### Issue: Out of memory
```bash
# Check memory
free -h

# Solutions:
# 1. Resize instance (upgrade to 16GB RAM)
# 2. Reduce workers in systemd service (from 2 to 1)
# 3. Add swap:
fallocate -l 4G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

---

## Cost Estimate

### Development/Testing Setup (Single Instance)
- ECS c7.xlarge: ¥0.6/hour = ¥432/month
- Public IP (5Mbps): ¥0.03/hour = ¥22/month
- OSS Storage (5GB): ¥0.6/month
- **Total: ~¥455/month (~$65 USD)**

### Ways to Save
- **Use Reserved Instances**: 30-60% discount for 1-year commitment
- **Stop when not in use**: Pay only for running hours
- **Use Preemptible Instances**: 80% discount (for non-critical workloads)
- **Downgrade to smaller instance**: ecs.c7.large (2 vCPU, 4GB) = ¥0.3/hour

---

## Next Steps

1. **Secure your API**:
   - Add authentication (API keys)
   - Setup HTTPS with SSL certificate
   - Restrict security group to specific IPs

2. **Monitor performance**:
   - Enable CloudMonitor
   - Setup alerts for high CPU/memory
   - Track request latency

3. **Scale up**:
   - Add more instances behind load balancer
   - Enable auto-scaling
   - Setup backup/disaster recovery

4. **Optimize costs**:
   - Analyze usage patterns
   - Schedule stop/start for off-hours
   - Consider reserved instances

---

## Resources

- **Full Deployment Guide**: [ALIBABA_CLOUD_DEPLOYMENT.md](ALIBABA_CLOUD_DEPLOYMENT.md)
- **Alibaba Cloud Console**: https://homenew.console.aliyun.com
- **ECS Documentation**: https://www.alibabacloud.com/help/ecs
- **OSS Documentation**: https://www.alibabacloud.com/help/oss

---

## Support

If you encounter issues:
1. Check logs: `journalctl -u linguabridge-api -n 50`
2. Review [ALIBABA_CLOUD_DEPLOYMENT.md](ALIBABA_CLOUD_DEPLOYMENT.md) troubleshooting section
3. Alibaba Cloud Support: https://workorder.console.aliyun.com

Good luck! 🚀
