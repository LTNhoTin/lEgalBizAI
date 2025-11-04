# 🚀 Quick Start Guide

Hướng dẫn nhanh để bắt đầu fine-tune GPT-OSS 20B.

## 1️⃣ Setup (chỉ cần làm 1 lần)

```bash
# Di chuyển vào thư mục
cd /home/nhotin/work/LegalBizAI_project/finetune/gpt_oss_20b_finetune

# Chạy setup script
./setup.sh

# Hoặc manual setup:
pip install -r requirements.txt
wandb login  # Optional
```

## 2️⃣ Kiểm tra Environment

```bash
# Check xem mọi thứ đã OK chưa
python check_env.py
```

Nếu tất cả đều ✅ thì có thể tiếp tục.

## 3️⃣ Test Training (khuyến nghị)

```bash
# Chạy training với config nhỏ để test (~15-30 phút)
python main.py --test
```

Config test mode:
- 100 samples
- LoRA rank 8
- Max length 512
- 1 epoch

## 4️⃣ Full Training

```bash
# Chạy training đầy đủ (~4-8 giờ)
python main.py
```

Config full mode:
- 2,000 samples (80% train, 20% test)
- LoRA rank 64
- Max length 2048
- 3 epochs

## 5️⃣ Monitor Training

### Wandb Dashboard
Mở: https://wandb.ai/your-username/gpt-oss-20b-legal-finetune

### Local Logs
```bash
# Xem logs real-time
tail -f logs/training_*.log
```

### GPU Usage
```bash
# Monitor GPU
watch -n 1 nvidia-smi
```

## 6️⃣ Test Model

```bash
# Interactive mode
python inference.py

# Single question
python inference.py --question "Câu hỏi của bạn"

# Với custom model path
python inference.py --model-path outputs/final_model
```

## ⚙️ Customize Config

Edit `config/config.py`:

```python
# Giảm memory usage
per_device_train_batch_size: 1
gradient_accumulation_steps: 32  # Tăng
max_length: 1024  # Giảm

# Giảm LoRA rank
r: 32  # Hoặc 16

# Tăng tốc training (giảm quality)
num_train_epochs: 2
gradient_checkpointing: False
```

## 🔥 Common Issues

### Out of Memory
```python
# Trong config.py, giảm:
max_length: 1024  # Hoặc 512
r: 32  # Hoặc 16
gradient_accumulation_steps: 32  # Tăng
```

### Slow Training
- Check GPU usage: `nvidia-smi`
- Enable bf16: `bf16: True`
- Enable gradient checkpointing: `gradient_checkpointing: True`

### Wandb Issues
```bash
# Training không cần wandb
python main.py --no-wandb
```

## 📊 Expected Results

### Test Mode (~30 phút)
- Samples: 100
- Loss: ~2.5-3.0
- Output: `outputs_test/`

### Full Mode (~6 giờ)
- Samples: 2,000
- Loss: ~1.5-2.0
- Output: `outputs/`

## 💡 Tips

1. **Luôn chạy test mode trước** để đảm bảo setup OK
2. **Monitor GPU VRAM** với `nvidia-smi`
3. **Save checkpoints** để có thể resume nếu bị gián đoạn
4. **Use wandb** để track experiments dễ dàng
5. **Tune hyperparameters** dựa trên validation loss

## 🎯 Next Steps

Sau khi training xong:

1. Test model: `python inference.py`
2. Evaluate metrics
3. Tune hyperparameters nếu cần
4. Train lại với config tốt hơn
5. Deploy model vào production

## 📚 More Info

- Full documentation: `README.md`
- Config details: `config/config.py`
- Troubleshooting: `README.md#troubleshooting`

## ❓ Need Help?

1. Check logs: `logs/training_*.log`
2. Check wandb dashboard
3. Run: `python check_env.py`
4. See README.md for detailed docs

---

Happy Training! 🚀

