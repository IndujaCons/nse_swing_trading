#!/bin/bash
# ============================================================
# S3 Backup — RS Screener data_store/
# Runs daily at 16:00 IST via cron
#
# Setup:
#   1. Set RS_S3_BUCKET env var (or edit BUCKET below)
#   2. Ensure AWS CLI is configured (aws configure)
#   3. Enable versioning on bucket:
#        aws s3api put-bucket-versioning \
#          --bucket $RS_S3_BUCKET \
#          --versioning-configuration Status=Enabled
#   4. Add to crontab (crontab -e):
#        0 16 * * 1-5 /path/to/scripts/backup_data.sh >> /path/to/logs/backup.log 2>&1
#
# Restore:
#   aws s3 sync s3://$RS_S3_BUCKET/rs-screener/data_store/ /path/to/data_store/
# ============================================================

BUCKET="${RS_S3_BUCKET:-rs-screener-data}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOCAL_PATH="$PROJECT_DIR/data_store/"
S3_PATH="s3://${BUCKET}/rs-screener/data_store/"

TS="$(date '+%Y-%m-%d %H:%M:%S IST')"

echo "[$TS] Starting backup: $LOCAL_PATH → $S3_PATH"

if ! command -v aws &>/dev/null; then
    echo "[$TS] ERROR: aws CLI not found"
    exit 1
fi

aws s3 sync "$LOCAL_PATH" "$S3_PATH" \
    --delete \
    --exclude "*.pyc" \
    --exclude "__pycache__/*" \
    --exclude "*.tmp"

STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "[$TS] Backup complete ✓"
else
    echo "[$TS] Backup FAILED (exit $STATUS)"
fi
exit $STATUS
