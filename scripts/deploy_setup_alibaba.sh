#!/bin/bash
# Alibaba Cloud ECS Setup Script
# Run this on your ECS instance after first launch
# Usage: bash deploy_setup_alibaba.sh

set -e  # Exit on error

echo "=========================================="
echo "LinguaBridge - Alibaba Cloud Setup"
echo "=========================================="
echo ""

# Configuration
PROJECT_DIR="/opt/linguabridge"
PYTHON_VERSION="3.10"
OSS_BUCKET="linguabridge-models-CHANGE_THIS"  # Change to your bucket name

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    error "Please run as root (sudo bash deploy_setup_alibaba.sh)"
fi

# Step 1: Update system
info "Step 1: Updating system packages..."
apt update
apt upgrade -y

# Step 2: Install dependencies
info "Step 2: Installing system dependencies..."
apt install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    git \
    curl \
    wget \
    vim \
    htop \
    net-tools

# Step 3: Install ossutil
info "Step 3: Installing ossutil..."
if [ ! -f /usr/local/bin/ossutil ]; then
    wget http://gosspublic.alicdn.com/ossutil/1.7.18/ossutil64 -O /tmp/ossutil64
    chmod 755 /tmp/ossutil64
    mv /tmp/ossutil64 /usr/local/bin/ossutil
    info "ossutil installed. Please run 'ossutil config' to configure."
else
    info "ossutil already installed"
fi

# Step 4: Create project directory
info "Step 4: Creating project directory..."
mkdir -p ${PROJECT_DIR}
cd ${PROJECT_DIR}

# Step 5: Setup Git (if repo URL provided)
if [ -n "$1" ]; then
    info "Step 5: Cloning repository from $1..."
    if [ -d "${PROJECT_DIR}/.git" ]; then
        warn "Git repository already exists, skipping clone"
    else
        git clone "$1" ${PROJECT_DIR}
    fi
else
    warn "Step 5: No Git repository URL provided, please upload code manually"
    info "You can use: scp -r /local/path root@${HOSTNAME}:${PROJECT_DIR}/"
fi

# Step 6: Create virtual environment
info "Step 6: Creating Python virtual environment..."
if [ ! -d "${PROJECT_DIR}/.venv" ]; then
    python${PYTHON_VERSION} -m venv ${PROJECT_DIR}/.venv
    info "Virtual environment created"
else
    info "Virtual environment already exists"
fi

# Activate virtual environment
source ${PROJECT_DIR}/.venv/bin/activate

# Step 7: Install Python dependencies
info "Step 7: Installing Python dependencies..."
pip install --upgrade pip

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    info "GPU detected, installing PaddlePaddle GPU version..."
    pip install paddlepaddle-gpu==3.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
else
    info "No GPU detected, installing PaddlePaddle CPU version..."
    pip install paddlepaddle==3.0.0
fi

# Install other requirements
if [ -f "${PROJECT_DIR}/requirements.txt" ]; then
    pip install -r ${PROJECT_DIR}/requirements.txt
else
    warn "requirements.txt not found, skipping"
fi

# Step 8: Create necessary directories
info "Step 8: Creating project directories..."
mkdir -p ${PROJECT_DIR}/{data/{raw,processed,soft_labels},models/{teacher,student},logs,cache}

# Step 9: Download models from OSS (if configured)
info "Step 9: Checking OSS configuration..."
if ossutil ls oss://${OSS_BUCKET} &> /dev/null; then
    info "OSS configured successfully"
    read -p "Download models from OSS? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        info "Downloading student model from OSS..."
        ossutil cp -r oss://${OSS_BUCKET}/models/student/ ${PROJECT_DIR}/models/student/ --recursive || warn "Model download failed or bucket empty"
    fi
else
    warn "OSS not configured. Please run: ossutil config"
fi

# Step 10: Create systemd service
info "Step 10: Creating systemd service..."
cat > /etc/systemd/system/linguabridge-api.service << EOF
[Unit]
Description=LinguaBridge Translation API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=${PROJECT_DIR}
Environment="PATH=${PROJECT_DIR}/.venv/bin"
ExecStart=${PROJECT_DIR}/.venv/bin/python -m uvicorn src.app_api:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
info "Systemd service created"

# Step 11: Configure firewall (if needed)
info "Step 11: Checking firewall..."
if command -v ufw &> /dev/null; then
    ufw --force disable
    info "UFW firewall disabled (use Alibaba Security Groups instead)"
fi

# Step 12: Display summary
echo ""
echo "=========================================="
info "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Configure ossutil if not done: ossutil config"
echo "2. Download models: ossutil cp -r oss://${OSS_BUCKET}/models/student/ models/student/"
echo "3. Start API service: systemctl start linguabridge-api"
echo "4. Check status: systemctl status linguabridge-api"
echo "5. View logs: journalctl -u linguabridge-api -f"
echo "6. Test API: curl http://localhost:8000/health"
echo ""
echo "To enable service on boot: systemctl enable linguabridge-api"
echo ""

# Optional: Start service
read -p "Start API service now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "${PROJECT_DIR}/src/app_api.py" ]; then
        systemctl start linguabridge-api
        sleep 3
        systemctl status linguabridge-api --no-pager
        echo ""
        info "Testing API..."
        sleep 2
        curl -f http://localhost:8000/health && info "API is healthy!" || warn "API health check failed"
    else
        warn "app_api.py not found, cannot start service"
    fi
fi

echo ""
info "All done! 🚀"
