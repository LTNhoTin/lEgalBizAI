#!/bin/bash
# Script to renew Let's Encrypt SSL certificate
# This script should be run via cron (e.g., weekly)

set -e

COMPOSE_FILE="/home/nhotin/work/LegalBizAI_project/docker-compose.yml"
DOMAIN="legalbizai.hypersona.vn"

echo "Starting SSL certificate renewal for $DOMAIN..."

# Stop nginx container to free port 80
echo "Stopping nginx container..."
cd /home/nhotin/work/LegalBizAI_project
docker-compose stop nginx

# Renew certificate
echo "Renewing certificate..."
sudo certbot renew --standalone --non-interactive

# Reload nginx configuration
echo "Restarting nginx container..."
docker-compose up -d nginx

echo "SSL certificate renewal completed successfully!"


