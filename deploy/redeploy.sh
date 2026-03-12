#!/bin/bash
# Quick redeploy on EC2 after code changes
# Run from your Mac: bash deploy/redeploy.sh <EC2_IP>

EC2_IP=$1
if [ -z "$EC2_IP" ]; then
    echo "Usage: bash deploy/redeploy.sh <EC2_PUBLIC_IP>"
    exit 1
fi

echo "Syncing code to EC2..."
rsync -avz --exclude='venv/' --exclude='__pycache__/' --exclude='.git/' --exclude='.env' \
    . ubuntu@${EC2_IP}:/home/ubuntu/relative_strength/

echo "Restarting service..."
ssh ubuntu@${EC2_IP} "sudo systemctl restart rs-dashboard"

echo "Done! http://${EC2_IP}:8080"
