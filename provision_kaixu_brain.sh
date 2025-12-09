#!/bin/bash
# save as: provision_kaixu_brain.sh
# Run on fresh Ubuntu 22.04 cloud GPU instance
# GPU Requirements: RTX 4090 (24GB+) / RTX 5090 (32GB+)

set -e  # Exit on any error

echo "=== KAIXU CLOUD BRAIN v1 PROVISIONING ==="

# Update system
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Python 3.11
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
sudo update-alternatives --set python3 /usr/bin/python3.11

# Install CUDA toolkit (if not pre-installed)
sudo apt-get install -y nvidia-cuda-toolkit nvidia-driver-535

# Install system dependencies
sudo apt-get install -y git curl wget build-essential libssl-dev libffi-dev
sudo apt-get install -y htop nvtop screen tmux

# Create dedicated user for Kaixu
sudo useradd -m -s /bin/bash kaixu
sudo usermod -aG sudo kaixu
echo "kaixu ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/kaixu

# Switch to kaixu user
sudo -u kaixu bash << 'EOF'
cd /home/kaixu

# Create virtual environment
python3.11 -m venv kaixu-venv
source kaixu-venv/bin/activate

# Install vLLM with CUDA 12.1 support
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "vllm==0.3.3" fastapi uvicorn
pip install huggingface-hub python-dotenv aiohttp

# Create directory structure
mkdir -p /home/kaixu/kaixu-brain/{logs,models,cache,config}

# Download Llama 3.1 8B Instruct model
export HF_TOKEN="YOUR_ACTUAL_HUGGINGFACE_TOKEN_HERE"  # REPLACE WITH ACTUAL TOKEN
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'meta-llama/Llama-3.1-8B-Instruct',
    local_dir='/home/kaixu/kaixu-brain/models/llama-3.1-8b-instruct',
    token='$HF_TOKEN',
    ignore_patterns=['*.safetensors', '*.bin'],
    max_workers=4
)
"

# Create systemd service file
sudo tee /etc/systemd/system/kaixu-brain.service << 'SERVICE'
[Unit]
Description=Kaixu Cloud Brain v1 - 8B LLM Service
After=network.target

[Service]
User=kaixu
Group=kaixu
WorkingDirectory=/home/kaixu
Environment="PATH=/home/kaixu/kaixu-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="HF_TOKEN=YOUR_ACTUAL_HUGGINGFACE_TOKEN_HERE"
ExecStart=/home/kaixu/kaixu-venv/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model /home/kaixu/kaixu-brain/models/llama-3.1-8b-instruct \
    --port 8000 \
    --host 0.0.0.0 \
    --api-key kaixu-internal-key \
    --served-model-name kaixu-brain-v1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --enforce-eager \
    --disable-log-requests \
    --log-level info
Restart=always
RestartSec=10
StandardOutput=append:/home/kaixu/kaixu-brain/logs/vllm.log
StandardError=append:/home/kaixu/kaixu-brain/logs/vllm-error.log

[Install]
WantedBy=multi-user.target
SERVICE

# Create startup script
cat > /home/kaixu/start_kaixu.sh << 'STARTUP'
#!/bin/bash
source /home/kaixu/kaixu-venv/bin/activate

# Start vLLM server
nohup python -m vllm.entrypoints.openai.api_server \
    --model /home/kaixu/kaixu-brain/models/llama-3.1-8b-instruct \
    --port 8000 \
    --host 0.0.0.0 \
    --api-key kaixu-internal-key \
    --served-model-name kaixu-brain-v1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    > /home/kaixu/kaixu-brain/logs/vllm.log 2>&1 &

echo "Kaixu Brain v1 started on port 8000"
echo "Monitor logs: tail -f /home/kaixu/kaixu-brain/logs/vllm.log"
STARTUP

chmod +x /home/kaixu/start_kaixu.sh

# Create health check script
cat > /home/kaixu/health_check.py << 'HEALTH'
#!/usr/bin/env python3
import requests
import json
import sys

def check_brain():
    try:
        # Check models endpoint
        resp = requests.get("http://127.0.0.1:8000/v1/models", timeout=10)
        if resp.status_code == 200:
            models = resp.json()
            if "data" in models and any("kaixu" in m.get("id", "").lower() for m in models["data"]):
                print("✓ Kaixu Brain v1 is running")
                return True
        
        # Try chat completion
        resp = requests.post(
            "http://127.0.0.1:8000/v1/chat/completions",
            json={
                "model": "kaixu-brain-v1",
                "messages": [{"role": "user", "content": "Say OK if operational."}],
                "max_tokens": 10
            },
            timeout=30
        )
        if resp.status_code == 200:
            print("✓ Kaixu Brain v1 responding to requests")
            return True
        else:
            print(f"✗ Brain returned status {resp.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

if __name__ == "__main__":
    if check_brain():
        sys.exit(0)
    else:
        sys.exit(1)
HEALTH

chmod +x /home/kaixu/health_check.py

echo "=== PROVISIONING COMPLETE ==="
echo "1. Start service: sudo systemctl start kaixu-brain"
echo "2. Enable auto-start: sudo systemctl enable kaixu-brain"
echo "3. Check status: sudo systemctl status kaixu-brain"
echo "4. Test endpoint: curl http://localhost:8000/v1/models"
EOF

# Reload systemd and start service
sudo systemctl daemon-reload
sudo systemctl start kaixu-brain
sudo systemctl enable kaixu-brain

echo "=== INSTALLATION VERIFICATION ==="
sleep 10
sudo -u kaixu python3 /home/kaixu/health_check.py

if [ $? -eq 0 ]; then
    echo "✅ Kaixu Cloud Brain v1 provisioned successfully"
    echo "📡 API available at: http://$(curl -s ifconfig.me):8000/v1/chat/completions"
    echo "🔑 API Key: kaixu-internal-key"
else
    echo "❌ Provisioning failed. Check logs: sudo journalctl -u kaixu-brain -f"
Fi
