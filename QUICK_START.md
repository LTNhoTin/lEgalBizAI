# Quick Start - LegalBizAI Deployment

## Các bước triển khai nhanh

### 1. Đảm bảo Backend đang chạy
```bash
# Chạy backend FastAPI trên port 8000
cd /home/nhotin/work/LegalBizAI_project/backend
# uvicorn main:app --host 127.0.0.1 --port 8000
```

### 2. Dừng Nginx hệ thống (nếu đang chạy)
```bash
sudo systemctl stop nginx
```

### 3. Khởi động Docker Nginx
```bash
cd /home/nhotin/work/LegalBizAI_project
docker-compose up -d
```

### 4. Kiểm tra logs
```bash
docker-compose logs -f nginx
```

### 5. Kiểm tra hoạt động
```bash
# Health check
curl http://localhost/health

# Frontend
curl http://localhost/LegalbizAi_chatbot

# Domain (nếu DNS đã trỏ về)
curl http://legalbizai.hypersona.vn/LegalbizAi_chatbot
```

## Các file đã được tạo

- `nginx/nginx.conf` - Cấu hình nginx chính
- `nginx/legalbizai.conf` - Cấu hình site cho legalbizai.hypersona.vn
- `docker-compose.yml` - Docker compose configuration
- `DEPLOY.md` - Hướng dẫn chi tiết

## Lưu ý

- Backend phải chạy trên host machine (127.0.0.1:8000)
- Frontend đã được build vào `frontend/dist/`
- API calls từ frontend sẽ được proxy qua `/api/` → backend

## Troubleshooting

Nếu có lỗi, xem file `DEPLOY.md` để biết chi tiết troubleshooting.


