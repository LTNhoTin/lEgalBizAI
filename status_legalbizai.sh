#!/bin/bash

###############################################################################
# LegalBizAI Status Script
# Kiểm tra trạng thái của tất cả services
###############################################################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         📊 LegalBizAI System Status                         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to check service status
check_service() {
    local service_name=$1
    local port=$2
    local url=$3
    
    printf "%-20s " "$service_name:"
    
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "${GREEN}✓ Running${NC} (port $port)"
        if [ ! -z "$url" ]; then
            echo "                      URL: $url"
        fi
        return 0
    else
        echo -e "${RED}✗ Stopped${NC} (port $port)"
        return 1
    fi
}

# Check Backend
echo -e "${BLUE}Backend Service:${NC}"
check_service "  FastAPI" 8000 "http://localhost:8000"

# Check Frontend
echo ""
echo -e "${BLUE}Frontend Service:${NC}"
check_service "  Vite Dev Server" 5173 "http://localhost:5173"

# Check Nginx
echo ""
echo -e "${BLUE}Reverse Proxy:${NC}"
printf "%-20s " "  Nginx:"
if sudo systemctl is-active --quiet nginx; then
    echo -e "${GREEN}✓ Running${NC}"
    echo "                      Public: https://legalbizai.hypersona.vn"
else
    echo -e "${RED}✗ Stopped${NC}"
fi

# Check Ollama
echo ""
echo -e "${BLUE}AI Model Service:${NC}"
check_service "  Ollama" 11434 "http://localhost:11434"

# Check SSL Certificate
echo ""
echo -e "${BLUE}SSL Certificate:${NC}"
printf "%-20s " "  Let's Encrypt:"
if sudo certbot certificates 2>/dev/null | grep -q "legalbizai.hypersona.vn"; then
    EXPIRY=$(sudo certbot certificates 2>/dev/null | grep "Expiry Date" | head -1 | awk '{print $3, $4}')
    echo -e "${GREEN}✓ Valid${NC}"
    echo "                      Expires: $EXPIRY"
else
    echo -e "${YELLOW}⚠ Not configured${NC}"
fi

# System Resources
echo ""
echo -e "${BLUE}System Resources:${NC}"
echo "  CPU Usage:  $(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')"
echo "  Memory:     $(free -h | awk '/^Mem:/ {print $3 " / " $2 " (" int($3/$2 * 100) "%)"}')"
echo "  Disk:       $(df -h / | awk 'NR==2 {print $3 " / " $2 " (" $5 ")"}')"

# Recent Logs
echo ""
echo -e "${BLUE}Quick Health Check:${NC}"
printf "  Backend API:     "
if curl -s http://localhost:8000/docs > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Responding${NC}"
else
    echo -e "${RED}✗ Not responding${NC}"
fi

printf "  Frontend:        "
if curl -s http://localhost:5173 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Responding${NC}"
else
    echo -e "${RED}✗ Not responding${NC}"
fi

printf "  HTTPS:           "
if curl -s -k https://legalbizai.hypersona.vn/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Responding${NC}"
else
    echo -e "${RED}✗ Not responding${NC}"
fi

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Quick Commands:                                            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo "  Start:   ./start_legalbizai.sh"
echo "  Stop:    ./stop_legalbizai.sh"
echo "  Status:  ./status_legalbizai.sh"
echo "  Logs:    tail -f backend/backend.log"
echo ""
