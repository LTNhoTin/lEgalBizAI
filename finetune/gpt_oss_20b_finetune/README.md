# GPT-OSS 20B Fine-tuning với LoRA/QLoRA

Fine-tune GPT-OSS 20B model trên Vietnamese Legal QA dataset với LoRA/QLoRA để fit vào GPU 16GB VRAM.

## 📋 Yêu cầu

- **GPU**: NVIDIA GPU với ít nhất 16GB VRAM (RTX 3090, RTX 4090, A100, etc.)
- **CUDA**: CUDA 11.8+ hoặc 12.1+
- **Python**: Python 3.8+
- **RAM**: Ít nhất 32GB RAM khuyến nghị

## 🚀 Setup

### 1. Cài đặt dependencies

```bash
cd /home/nhotin/work/LegalBizAI_project/finetune/gpt_oss_20b_finetune
pip install -r requirements.txt
```

### 2. Setup Wandb (optional nhưng khuyến nghị)

```bash
# Login vào wandb
wandb login

# Hoặc set API key
export WANDB_API_KEY=your_api_key_here
```

### 3. Kiểm tra data

Data phải nằm ở: `../data_finetunning/qaset_full_article.json`

Format data:
```json
[
  {
    "question": "Câu hỏi...",
    "answer": "Câu trả lời...",
    "references": [...],
    "chunk_ids": [...],
    ...
  }
]
```

## 🎯 Cách sử dụng

### Training với full config

```bash
python main.py
```

Sẽ training với:
- **Data**: 2,000 samples (80% train, 20% test)
- **LoRA rank**: 64
- **Max length**: 2048 tokens
- **Epochs**: 3
- **Batch size**: 1 với gradient accumulation 16
- **Learning rate**: 2e-4
- **Quantization**: 4-bit (QLoRA)

### Test mode (training nhanh để test setup)

```bash
python main.py --test
```

Sẽ training với config nhỏ hơn:
- **Data**: 100 samples
- **LoRA rank**: 8
- **Max length**: 512 tokens
- **Epochs**: 1
- **Gradient accumulation**: 4

### Tắt Wandb logging

```bash
python main.py --no-wandb
```

## 📊 Config

### Model Config (`config/config.py`)

```python
ModelConfig:
    model_name: "pansophic/rocket-3B"  # Hoặc GPT-OSS 20B model name
    load_in_4bit: True                  # QLoRA 4-bit
    bnb_4bit_quant_type: "nf4"         # NF4 quantization
    bnb_4bit_use_double_quant: True    # Nested quantization
```

### LoRA Config

```python
LoRAConfig:
    r: 64                    # LoRA rank
    lora_alpha: 16          # Alpha parameter
    target_modules: [...]   # Modules to apply LoRA
    lora_dropout: 0.05      # Dropout
```

### Training Config

```python
TrainingConfig:
    num_train_epochs: 3
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 16
    learning_rate: 2e-4
    optim: "paged_adamw_32bit"
    bf16: True
    gradient_checkpointing: True
```

### Data Config

```python
DataConfig:
    max_samples: 2000       # Giảm từ 96k xuống 2k
    train_split: 0.8        # 80% train
    test_split: 0.2         # 20% test
    max_length: 2048        # Max tokens
```

## 📁 Cấu trúc thư mục

```
gpt_oss_20b_finetune/
├── config/
│   └── config.py          # Config file
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Data loading & processing
│   └── trainer.py         # Training logic
├── data/                  # Processed data cache
├── outputs/               # Training outputs
│   ├── checkpoint-*/      # Checkpoints
│   └── final_model/       # Final saved model
├── outputs_test/          # Test mode outputs
├── logs/                  # Training logs
├── main.py               # Main training script
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 🔧 Troubleshooting

### Out of Memory (OOM)

Nếu gặp OOM error, thử:

1. Giảm batch size hoặc tăng gradient accumulation:
```python
per_device_train_batch_size: 1
gradient_accumulation_steps: 32  # Tăng lên
```

2. Giảm max_length:
```python
max_length: 1024  # Hoặc 512
```

3. Giảm LoRA rank:
```python
r: 32  # Hoặc 16
```

### CUDA Error

Kiểm tra CUDA version:
```bash
nvidia-smi
python -c "import torch; print(torch.version.cuda)"
```

### Slow Training

- Đảm bảo đang dùng GPU (check với `nvidia-smi`)
- Enable `gradient_checkpointing=True`
- Enable `bf16=True` nếu GPU hỗ trợ

## 📈 Monitoring

### Wandb Dashboard

Xem training progress tại: https://wandb.ai/your-username/gpt-oss-20b-legal-finetune

Metrics được log:
- Training loss
- Evaluation loss
- Learning rate
- GPU memory usage
- Training speed (samples/sec)

### Local Logs

Logs được lưu tại: `logs/training_YYYYMMDD_HHMMSS.log`

## 💾 Saved Model

Model được lưu tại:
- Checkpoints: `outputs/checkpoint-*/`
- Final model: `outputs/final_model/`

Load model sau khi train:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("pansophic/rocket-3B")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "outputs/final_model")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("outputs/final_model")
```

## 🎓 Training Tips

1. **Luôn chạy test mode trước**: `python main.py --test` để đảm bảo mọi thứ hoạt động
2. **Monitor GPU usage**: Dùng `nvidia-smi` để theo dõi VRAM
3. **Save checkpoints thường xuyên**: Config mặc định save mỗi 200 steps
4. **Use wandb**: Giúp track experiments và compare results
5. **Tune hyperparameters**: Điều chỉnh learning rate, batch size dựa trên results

## 📝 Notes

- Training full config (~3 epochs, 2000 samples) có thể mất 4-8 giờ tùy GPU
- Test mode (~1 epoch, 100 samples) mất ~15-30 phút
- Model size sau khi train: ~100-200MB (chỉ LoRA weights)
- GPU VRAM usage: ~14-15GB với 4-bit quantization

## 🐛 Báo lỗi

Nếu gặp vấn đề, check logs tại `logs/` và wandb dashboard.

## 📚 Tài liệu tham khảo

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

