#!/bin/bash

###############################################################################
# LegalBizAI Startup Script
# Khởi động toàn bộ hệ thống: Backend, Frontend, Nginx
###############################################################################

set -e  # Exit on error

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
echo -e "${BLUE}║         🚀 LegalBizAI System Startup Script                 ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to check if a service is running
check_service() {
    local service_name=$1
    local port=$2
    
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "${GREEN}✓${NC} $service_name is already running on port $port"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} $service_name is not running on port $port"
        return 1
    fi
}

# Function to start backend
start_backend() {
    echo -e "\n${BLUE}[1/3] Starting Backend (FastAPI on port 8000)...${NC}"
    
    if check_service "Backend" 8000; then
        echo "Skipping backend startup"
    else
        echo "Starting backend with conda environment: legalbizai..."
        cd $BACKEND_DIR
        
        # Start backend with conda environment activated
        nohup bash -c "eval \"\$(conda shell.bash hook)\" && conda activate legalbizai && python main.py" > backend.log 2>&1 &
        echo $! > backend.pid
        sleep 3
        
        if check_service "Backend" 8000; then
            echo -e "${GREEN}✓ Backend started successfully${NC}"
        else
            echo -e "${RED}✗ Failed to start backend${NC}"
            exit 1
        fi
    fi
}

# Function to start frontend
start_frontend() {
    echo -e "\n${BLUE}[2/3] Starting Frontend (Vite on port 5173)...${NC}"
    
    if check_service "Frontend" 5173; then
        echo "Skipping frontend startup"
    else
        echo "Starting frontend..."
        cd $FRONTEND_DIR
        nohup npm run dev > frontend.log 2>&1 &
        echo $! > frontend.pid
        sleep 5
        
        if check_service "Frontend" 5173; then
            echo -e "${GREEN}✓ Frontend started successfully${NC}"
        else
            echo -e "${RED}✗ Failed to start frontend${NC}"
            exit 1
        fi
    fi
}

# Function to start/restart nginx
start_nginx() {
    echo -e "\n${BLUE}[3/3] Starting Nginx...${NC}"
    
    # Check if nginx is running
    if sudo systemctl is-active --quiet nginx; then
        echo "Nginx is already running, reloading configuration..."
        sudo systemctl reload nginx
        echo -e "${GREEN}✓ Nginx configuration reloaded${NC}"
    else
        echo "Starting nginx..."
        sudo systemctl start nginx
        if sudo systemctl is-active --quiet nginx; then
            echo -e "${GREEN}✓ Nginx started successfully${NC}"
        else
            echo -e "${RED}✗ Failed to start nginx${NC}"
            exit 1
        fi
    fi
}

# Function to check Ollama
check_ollama() {
    echo -e "\n${BLUE}[Optional] Checking Ollama service...${NC}"
    
    if check_service "Ollama" 11434; then
        echo -e "${GREEN}✓ Ollama is running (GPT-OSS-20B model available)${NC}"
    else
        echo -e "${YELLOW}⚠ Ollama is not running${NC}"
        echo "  LegalBizAI model won't work without Ollama"
        echo "  Only LegalBizAI Pro (Gemini) will be available"
    fi
}

# Main execution
main() {
    echo "Starting all services..."
    echo ""
    
    start_backend
    start_frontend
    start_nginx
    check_ollama
    
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              ✅ All services started successfully!          ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}🌐 Access URLs:${NC}"
    echo -e "  Public:  ${GREEN}https://legalbizai.hypersona.vn${NC}"
    echo -e "  Local:   ${GREEN}http://localhost:5173${NC}"
    echo ""
    echo -e "${BLUE}📊 Service Status:${NC}"
    echo -e "  Backend:  http://localhost:8000 (API + Docs)"
    echo -e "  Frontend: http://localhost:5173 (Vite Dev Server)"
    echo -e "  Nginx:    Port 80/443 (Reverse Proxy)"
    echo -e "  Ollama:   http://localhost:11434 (AI Model)"
    echo ""
    echo -e "${BLUE}📝 Logs:${NC}"
    echo -e "  Backend:  $BACKEND_DIR/backend.log"
    echo -e "  Frontend: $FRONTEND_DIR/frontend.log"
    echo -e "  Nginx:    /var/log/nginx/legalbizai_*.log"
    echo ""
    echo -e "${YELLOW}💡 Tips:${NC}"
    echo -e "  - View backend logs:  tail -f $BACKEND_DIR/backend.log"
    echo -e "  - View frontend logs: tail -f $FRONTEND_DIR/frontend.log"
    echo -e "  - Stop all services:  ./stop_legalbizai.sh"
    echo ""
}

# Run main function
main
