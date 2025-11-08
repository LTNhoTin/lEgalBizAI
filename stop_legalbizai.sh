#!/bin/bash

###############################################################################
# LegalBizAI Stop Script
# Dừng toàn bộ hệ thống: Backend, Frontend
###############################################################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PROJECT_DIR="/home/nhotin/work/LegalBizAI_project"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         🛑 LegalBizAI System Stop Script                    ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to stop process by PID file
stop_by_pid() {
    local service_name=$1
    local pid_file=$2
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat $pid_file)
        if ps -p $pid > /dev/null 2>&1; then
            echo "Stopping $service_name (PID: $pid)..."
            kill $pid
            sleep 2
            if ps -p $pid > /dev/null 2>&1; then
                echo "Force killing $service_name..."
                kill -9 $pid
            fi
            rm $pid_file
            echo -e "${GREEN}✓ $service_name stopped${NC}"
        else
            echo -e "${YELLOW}⚠ $service_name is not running${NC}"
            rm $pid_file
        fi
    else
        echo -e "${YELLOW}⚠ No PID file found for $service_name${NC}"
    fi
}

# Function to stop process by name
stop_by_name() {
    local service_name=$1
    local process_pattern=$2
    
    echo "Stopping $service_name..."
    pkill -f "$process_pattern" 2>/dev/null
    
    sleep 2
    if pgrep -f "$process_pattern" > /dev/null; then
        echo "Force stopping $service_name..."
        pkill -9 -f "$process_pattern" 2>/dev/null
    fi
    
    if ! pgrep -f "$process_pattern" > /dev/null; then
        echo -e "${GREEN}✓ $service_name stopped${NC}"
    else
        echo -e "${RED}✗ Failed to stop $service_name${NC}"
    fi
}

# Stop backend
echo -e "${BLUE}[1/2] Stopping Backend...${NC}"
cd $BACKEND_DIR
if [ -f "backend.pid" ]; then
    stop_by_pid "Backend" "backend.pid"
else
    stop_by_name "Backend" "python main.py"
fi

# Stop frontend
echo -e "\n${BLUE}[2/2] Stopping Frontend...${NC}"
cd $FRONTEND_DIR
if [ -f "frontend.pid" ]; then
    stop_by_pid "Frontend" "frontend.pid"
else
    stop_by_name "Frontend" "vite --host"
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              ✅ All services stopped successfully!          ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}💡 Note:${NC}"
echo -e "  - Nginx is still running (system service)"
echo -e "  - Ollama is still running (if started)"
echo -e "  - To stop nginx: sudo systemctl stop nginx"
echo ""
echo -e "${BLUE}🔄 To restart:${NC} ./start_legalbizai.sh"
echo ""
