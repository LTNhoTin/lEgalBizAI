#!/bin/bash

###############################################################################
# LegalBizAI Nginx Startup Script
# Khởi động/Reload Nginx sau khi Backend và Frontend đã chạy
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         🌐 LegalBizAI Nginx Startup Script                  ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to check if a service is running
check_service() {
    local service_name=$1
    local port=$2
    
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "${GREEN}✓${NC} $service_name is running on port $port"
        return 0
    else
        echo -e "${RED}✗${NC} $service_name is NOT running on port $port"
        return 1
    fi
}

# Check prerequisites
echo -e "${BLUE}[Step 1/3] Checking prerequisites...${NC}"
echo ""

backend_running=true
frontend_running=true

if ! check_service "Backend" 8000; then
    backend_running=false
fi

if ! check_service "Frontend" 5173; then
    frontend_running=false
fi

echo ""

# Warning if services are not running
if [ "$backend_running" = false ] || [ "$frontend_running" = false ]; then
    echo -e "${YELLOW}⚠ WARNING: Some required services are not running!${NC}"
    echo ""
    
    if [ "$backend_running" = false ]; then
        echo -e "${YELLOW}  Backend is not running. To start it:${NC}"
        echo -e "  cd /home/nhotin/work/LegalBizAI_project/backend"
        echo -e "  nohup bash -c \"eval \\\"\\\$(conda shell.bash hook)\\\" && conda activate legalbizai && python main.py\" > backend.log 2>&1 &"
        echo ""
    fi
    
    if [ "$frontend_running" = false ]; then
        echo -e "${YELLOW}  Frontend is not running. To start it:${NC}"
        echo -e "  cd /home/nhotin/work/LegalBizAI_project/frontend"
        echo -e "  nohup npm run dev > frontend.log 2>&1 &"
        echo ""
    fi
    
    read -p "$(echo -e ${YELLOW}Do you want to continue starting Nginx anyway? [y/N]: ${NC})" -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Aborted.${NC}"
        exit 1
    fi
    echo ""
fi

# Test nginx configuration
echo -e "${BLUE}[Step 2/3] Testing Nginx configuration...${NC}"
if sudo nginx -t 2>&1 | grep -q "test is successful"; then
    echo -e "${GREEN}✓ Nginx configuration is valid${NC}"
else
    echo -e "${RED}✗ Nginx configuration has errors!${NC}"
    sudo nginx -t
    exit 1
fi
echo ""

# Start/Reload Nginx
echo -e "${BLUE}[Step 3/3] Starting/Reloading Nginx...${NC}"

if sudo systemctl is-active --quiet nginx; then
    echo "Nginx is already running, reloading configuration..."
    sudo systemctl reload nginx
    
    if sudo systemctl is-active --quiet nginx; then
        echo -e "${GREEN}✓ Nginx configuration reloaded successfully${NC}"
    else
        echo -e "${RED}✗ Failed to reload Nginx${NC}"
        exit 1
    fi
else
    echo "Starting Nginx..."
    sudo systemctl start nginx
    
    if sudo systemctl is-active --quiet nginx; then
        echo -e "${GREEN}✓ Nginx started successfully${NC}"
    else
        echo -e "${RED}✗ Failed to start Nginx${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              ✅ Nginx is now running!                       ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}🌐 Access URLs:${NC}"
echo -e "  Public:  ${GREEN}https://legalbizai.hypersona.vn${NC}"
echo -e "  Local:   ${GREEN}http://localhost${NC}"
echo ""
echo -e "${BLUE}📊 Nginx Status:${NC}"
echo -e "  Port 80:  HTTP (redirects to HTTPS)"
echo -e "  Port 443: HTTPS (SSL/TLS)"
echo ""
echo -e "${BLUE}🔍 Proxy Configuration:${NC}"
echo -e "  /api/*     → Backend  (http://localhost:8000)"
echo -e "  /*         → Frontend (http://localhost:5173)"
echo ""
echo -e "${BLUE}📝 Logs:${NC}"
echo -e "  Access: /var/log/nginx/legalbizai_access.log"
echo -e "  Error:  /var/log/nginx/legalbizai_error.log"
echo ""
echo -e "${YELLOW}💡 Useful commands:${NC}"
echo -e "  - View access logs: sudo tail -f /var/log/nginx/legalbizai_access.log"
echo -e "  - View error logs:  sudo tail -f /var/log/nginx/legalbizai_error.log"
echo -e "  - Check status:     sudo systemctl status nginx"
echo -e "  - Reload config:    sudo systemctl reload nginx"
echo -e "  - Restart nginx:    sudo systemctl restart nginx"
echo -e "  - Stop nginx:       sudo systemctl stop nginx"
echo ""


