# LegalBizAI Deployment Guide

## Tổng quan

Hướng dẫn này giúp bạn triển khai LegalBizAI lên production với Docker Nginx.

## Cấu trúc dự án

```
/home/nhotin/work/LegalBizAI_project/
├── docker-compose.yml          # Docker compose cho nginx
├── nginx/
│   ├── nginx.conf              # Main nginx config
│   └── legalbizai.conf         # Site config cho legalbizai.hypersona.vn
├── frontend/
│   └── dist/                   # Production build (sau khi build)
└── backend/
    └── main.py                 # FastAPI backend (chạy trên port 8000)
```

## Yêu cầu

- Docker và Docker Compose đã được cài đặt
- Backend FastAPI đang chạy trên port 8000 của host
- Domain `legalbizai.hypersona.vn` đã trỏ về IP server

## Các bước triển khai

### 1. Đảm bảo Backend đang chạy

Backend FastAPI cần chạy trên port 8000 của host machine:

```bash
cd /home/nhotin/work/LegalBizAI_project/backend
# Chạy backend (ví dụ: uvicorn main:app --host 127.0.0.1 --port 8000)
```

### 2. Build Frontend Production

Frontend đã được build. Nếu cần build lại:

```bash
cd /home/nhotin/work/LegalBizAI_project/frontend
npm install
npm run build
```

### 3. Tạm dừng Nginx hệ thống (nếu có)

```bash
sudo systemctl stop nginx
# Hoặc
sudo systemctl disable nginx  # Nếu không muốn tự động start
```

### 4. Chạy Docker Nginx

```bash
cd /home/nhotin/work/LegalBizAI_project
docker-compose up -d
```

### 5. Kiểm tra logs

```bash
# Xem logs của nginx container
docker-compose logs -f nginx

# Hoặc kiểm tra logs nginx trực tiếp
docker exec legalbizai_nginx cat /var/log/nginx/legalbizai_access.log
docker exec legalbizai_nginx cat /var/log/nginx/legalbizai_error.log
```

### 6. Kiểm tra hoạt động

```bash
# Kiểm tra health check
curl http://localhost/health

# Kiểm tra frontend
curl http://localhost/LegalbizAi_chatbot

# Kiểm tra domain (nếu DNS đã trỏ về)
curl http://legalbizai.hypersona.vn/LegalbizAi_chatbot
```

## Cấu hình

### Frontend
- Base path: `/LegalbizAi_chatbot`
- Build output: `frontend/dist/`
- API calls: `/api/stream` (sẽ được proxy đến backend)

### Backend
- URL: `http://127.0.0.1:8000` (host machine)
- Endpoint: `/stream`
- Nginx proxy: `/api/stream` → `http://host.docker.internal:8000/stream`

### Nginx
- Port: 80 (HTTP), 443 (HTTPS - cần cấu hình SSL)
- Domain: `legalbizai.hypersona.vn`
- Static files: `/usr/share/nginx/html` (mount từ `frontend/dist/`)

## Cấu hình SSL/HTTPS (Tùy chọn)

Để cấu hình HTTPS với Let's Encrypt:

1. Cài đặt certbot:
```bash
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx
```

2. Tạo certificate:
```bash
sudo certbot certonly --standalone -d legalbizai.hypersona.vn
```

3. Cập nhật `nginx/legalbizai.conf`:
   - Uncomment dòng redirect HTTP → HTTPS
   - Thêm cấu hình SSL server block

4. Mount certificates vào Docker container trong `docker-compose.yml`:
```yaml
volumes:
  - /etc/letsencrypt:/etc/letsencrypt:ro
```

5. Restart container:
```bash
docker-compose restart nginx
```

## Troubleshooting

### Backend không kết nối được

1. Kiểm tra backend có đang chạy:
```bash
curl http://127.0.0.1:8000/docs
```

2. Kiểm tra `host.docker.internal` có hoạt động:
```bash
docker exec legalbizai_nginx ping host.docker.internal
```

3. Nếu không hoạt động, thử dùng IP host thay vì `host.docker.internal`:
```bash
# Tìm IP host
ip addr show docker0 | grep inet
# Hoặc
hostname -I | awk '{print $1}'
```

### Frontend không load được

1. Kiểm tra file có được mount đúng:
```bash
docker exec legalbizai_nginx ls -la /usr/share/nginx/html
```

2. Kiểm tra permissions:
```bash
ls -la frontend/dist/
```

### Port 80 đã được sử dụng

1. Kiểm tra process nào đang dùng port 80:
```bash
sudo lsof -i :80
# hoặc
sudo netstat -tulpn | grep :80
```

2. Dừng service đang dùng port 80 hoặc đổi port trong `docker-compose.yml`

## Dừng và Xóa

```bash
# Dừng containers
docker-compose down

# Xóa containers và volumes
docker-compose down -v
```

## Cập nhật

Khi có thay đổi code:

1. Build lại frontend:
```bash
cd frontend && npm run build
```

2. Restart nginx container:
```bash
docker-compose restart nginx
```

## Lưu ý

- Backend phải chạy trên host machine (không phải trong Docker) để tránh vấn đề network
- Nếu backend chạy trong Docker, cần điều chỉnh cấu hình network trong docker-compose.yml
- Đảm bảo firewall mở port 80 và 443
- Kiểm tra DNS đã trỏ đúng về IP server


