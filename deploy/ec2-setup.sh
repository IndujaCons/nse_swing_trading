#!/bin/bash
# EC2 Setup — Run RS Dashboard as a systemd service
#
# 1. Launch an EC2 instance (Ubuntu 22.04, t3.small, ap-south-1)
# 2. Open port 8080 in security group
# 3. SSH in and run this script
#
# Usage: bash deploy/ec2-setup.sh

set -e

echo "======================================"
echo "  RS Dashboard — EC2 Setup"
echo "======================================"

# Install Python 3.13 and pip
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv git

# Clone repo (or scp your code)
cd /home/ubuntu
if [ ! -d "relative_strength" ]; then
    echo "Upload your code to /home/ubuntu/relative_strength first"
    echo "  scp -r . ubuntu@<EC2_IP>:/home/ubuntu/relative_strength/"
    exit 1
fi

cd relative_strength

# Create venv and install deps
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create systemd service
sudo tee /etc/systemd/system/rs-dashboard.service << 'SERVICE'
[Unit]
Description=RS Trading Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/relative_strength
Environment=FLASK_HOST=0.0.0.0
Environment=FLASK_PORT=8080
Environment=DEBUG_MODE=false
EnvironmentFile=/home/ubuntu/relative_strength/.env
ExecStart=/home/ubuntu/relative_strength/venv/bin/gunicorn --bind 0.0.0.0:8080 --workers 2 --threads 4 --timeout 300 ui.app:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable rs-dashboard
sudo systemctl start rs-dashboard

echo ""
echo "======================================"
echo "  Dashboard running on port 8080"
echo "  http://<EC2_PUBLIC_IP>:8080"
echo ""
echo "  Commands:"
echo "    sudo systemctl status rs-dashboard"
echo "    sudo systemctl restart rs-dashboard"
echo "    sudo journalctl -u rs-dashboard -f"
echo "======================================"
