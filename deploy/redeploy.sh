#!/bin/bash
# Quick redeploy on EC2 after code changes
# Run from your Mac: bash deploy/redeploy.sh <EC2_IP>

EC2_IP=$1
if [ -z "$EC2_IP" ]; then
    echo "Usage: bash deploy/redeploy.sh <EC2_PUBLIC_IP>"
    exit 1
fi

PEM_KEY="${HOME}/Downloads/strangle-key.pem"

echo "Syncing code to EC2..."
rsync -avz -e "ssh -i ${PEM_KEY}" \
    --exclude='venv/' --exclude='__pycache__/' --exclude='.git/' --exclude='.env' \
    . ubuntu@${EC2_IP}:/home/ubuntu/relative_strength/

echo "Restarting service..."
ssh -i "${PEM_KEY}" ubuntu@${EC2_IP} "sudo systemctl restart rs-dashboard"

echo "Done! http://${EC2_IP}:8080"
