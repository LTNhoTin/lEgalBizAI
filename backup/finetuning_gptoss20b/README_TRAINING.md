# GPT-OSS 20B Fine-tuning với Wandb Integration

## 🚀 Khởi chạy nhanh

### Cách 1: Tự động hoàn toàn
```bash
python run_training.py
```

### Cách 2: Chạy trực tiếp main.py
```bash
python main.py
```

## 📊 Metrics được theo dõi

### Training Metrics
- `train_loss` - Loss trong quá trình huấn luyện
- `learning_rate` - Tốc độ học hiện tại
- `epoch` - Epoch hiện tại
- `step` - Bước huấn luyện hiện tại

### Target Metrics (Mục tiêu)
- **Faithfulness**: ≥ 90% (% câu trả lời nhất quán với context)
- **Context Precision**: ≥ 85% (% nội dung câu trả lời dựa trên context)
- **Maliciousness**: ≤ 1% (% output được đánh dấu có hại)
- **ROUGE-L / BLEU**: ≥ 0.8 (Độ trùng lặp với ground truth)
- **Human Eval**: ≥ 4/5 (Điểm đánh giá thủ công về độ chính xác/rõ ràng)
- **Latency**: < 5s (Thời gian phản hồi trung bình)

### System Metrics
- GPU Memory Usage (Allocated/Cached)
- Training Time
- Pipeline Time

## 🔧 Cấu hình

### Wandb Configuration
```python
# Trong config.py
wandb:
  project_name: "gptoss-20b-finetune"
  run_name: "auto-generated"
  tags: ["gpt-oss", "20b", "finetune"]
  log_model: True
  watch_model: True
```

### Training Configuration
```python
# Các tham số huấn luyện chính
num_train_epochs: 3
per_device_train_batch_size: 1
learning_rate: 2e-4
logging_steps: 10
eval_steps: 100
```

## 📁 Cấu trúc thư mục

```
finetuning_gptoss20b/
├── main.py                 # Script chính
├── run_training.py         # Launcher tự động
├── setup_wandb.py         # Setup Wandb
├── config/
│   └── config.py          # Cấu hình toàn bộ
├── src/
│   ├── trainer.py         # GPTTrainer class
│   └── data_loader.py     # Data loading
└── outputs/               # Kết quả training
```

## 🐍 Conda Environment

Script sẽ tự động kích hoạt môi trường conda `gptoss finetune` nếu có.

Để tạo môi trường:
```bash
conda create -n "gptoss finetune" python=3.9
conda activate "gptoss finetune"
pip install -r requirements.txt
```

## 🧹 VRAM Management

Script tự động:
- Clear VRAM trước khi training
- Monitor GPU memory usage
- Clear VRAM sau khi training
- Log memory metrics lên Wandb

## 📊 Wandb Dashboard

Sau khi chạy, bạn có thể xem metrics tại:
- https://wandb.ai/your-username/gptoss-20b-finetune

### Metrics được log:
1. **Training Progress**: loss, learning rate, epoch
2. **GPU Usage**: memory allocated, cached, max usage
3. **Timing**: training time, pipeline time
4. **Model Performance**: so sánh với target metrics
5. **System Info**: environment, configuration

## 🔍 Troubleshooting

### Lỗi Conda Environment
```bash
# Nếu không tìm thấy environment
python main.py --skip-conda
```

### Lỗi Wandb
```bash
# Setup lại Wandb
python setup_wandb.py
```

### Lỗi VRAM
```bash
# Clear VRAM thủ công
python -c "import torch; torch.cuda.empty_cache()"
```

## 📝 Logs

Tất cả logs được lưu tại:
- Console output
- Wandb dashboard
- `outputs/` directory

## 🎯 Next Steps

Sau khi training hoàn thành:
1. Kiểm tra Wandb dashboard
2. Đánh giá model performance
3. Test với use cases cụ thể
4. Deploy nếu hài lòng với kết quả