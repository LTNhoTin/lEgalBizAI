# GPT-OSS 20B Finetuning Project

Dự án finetune model GPT-OSS 20B cho Sale Marketing với cấu trúc rõ ràng và dễ quản lý.

## 📁 Cấu trúc dự án

```
finetuning_gptoss20b/
├── config/
│   └── config.py              # Cấu hình model, training, data
├── src/
│   ├── data_loader.py         # Xử lý dataset
│   └── trainer.py             # Logic training
├── data/
│   └── sale_marketing_finetune_dataset.jsonl  # Dataset training
├── models/                    # Thư mục lưu model đã train
├── outputs/                   # Thư mục output training
├── logs/                      # Thư mục logs
├── scripts/                   # Scripts tiện ích
├── main.py                    # File chính để chạy training
├── inference.py               # File test model như ollama
├── requirements.txt           # Dependencies
└── README.md                  # Hướng dẫn này
```

## 🛠️ Cài đặt

### 1. Kích hoạt môi trường conda

```bash
conda activate gptoss
```

### 2. Cài đặt dependencies

```bash
# Cài đặt uv (package manager nhanh)
pip install --upgrade uv

# Cài đặt các dependencies chính
uv pip install torch>=2.8.0 triton>=3.4.0 numpy torchvision bitsandbytes datasets trl accelerate peft colorama tqdm psutil

# Cài đặt Unsloth (cần thiết cho finetuning)
uv pip install "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo"
uv pip install "unsloth[base] @ git+https://github.com/unslothai/unsloth"
uv pip install git+https://github.com/huggingface/transformers
uv pip install "git+https://github.com/triton-lang/triton.git@05b2c186c1b6c9a08375389d5efe9cb4c401c075#subdirectory=python/triton_kernels"
```

## Sử dụng

### 1. Training Model

#### Chạy training cơ bản:
```bash
python main.py
```

#### Chạy với tùy chọn:
```bash
# Thay đổi số epochs
python main.py --epochs 2

# Thay đổi batch size
python main.py --batch-size 2

# Thay đổi learning rate
python main.py --learning-rate 1e-4

# Đặt tên model tùy chỉnh
python main.py --model-name my_custom_model

# Bỏ qua setup checks
python main.py --skip-setup

# Quản lý cache
python main.py --cache-info                    # Hiển thị thông tin cache
python main.py --clear-cache all               # Xóa toàn bộ cache
python main.py --clear-cache model             # Chỉ xóa cache model
python main.py --clear-cache dataset           # Chỉ xóa cache dataset
python main.py --force-reload                  # Buộc tải lại mọi thứ (bỏ qua cache)
```

### 2. Test Model (Inference)

#### Chế độ interactive (giống ollama):
```bash
python inference.py
```

#### Test với prompt đơn:
```bash
python inference.py --prompt "Bên em dùng thử CRM 7 ngày rồi nhưng team chưa quen, giá gói Standard là bao nhiêu?"
```

#### Chọn persona khác:
```bash
# Sale Marketing (mặc định)
python inference.py --persona sale_marketing

# NhoTin - Lập trình viên
python inference.py --persona nhotin

# Thang Nguyen - Lập trình viên ngân hàng
python inference.py --persona thang_nguyen
```

#### Tùy chỉnh generation:
```bash
python inference.py --max-tokens 256 --temperature 0.8
```

### 3. Commands trong Interactive Mode

Khi chạy `python inference.py`, bạn có thể sử dụng các lệnh:

- `/clear` - Xóa lịch sử hội thoại
- `/history` - Hiển thị lịch sử hội thoại
- `/persona <name>` - Đổi persona (sale_marketing, nhotin, thang_nguyen)
- `/system <message>` - Đặt system message tùy chỉnh
- `/quit` - Thoát

## ⚙️ Cấu hình

Tất cả cấu hình được quản lý trong `config/config.py`:

### Model Config
- `model_name`: Tên model base (mặc định: "unsloth/gpt-oss-20b")
- `max_seq_length`: Độ dài sequence tối đa (mặc định: 1024)
- `lora_r`: Rank của LoRA (mặc định: 8)
- `lora_alpha`: Alpha của LoRA (mặc định: 16)

### Training Config
- `per_device_train_batch_size`: Batch size (mặc định: 1)
- `gradient_accumulation_steps`: Gradient accumulation (mặc định: 4)
- `num_train_epochs`: Số epochs (mặc định: 1)
- `learning_rate`: Learning rate (mặc định: 2e-4)

### Data Config
- `dataset_path`: Đường dẫn dataset (mặc định: "data/sale_marketing_finetune_dataset.jsonl")

## 💾 Cache System

Dự án sử dụng hệ thống cache thông minh để tăng tốc quá trình training:

### Tính năng Cache
- **Model Cache**: Lưu trữ model đã load và setup LoRA adapters
- **Dataset Cache**: Lưu trữ dataset đã được xử lý và format
- **Auto Detection**: Tự động phát hiện thay đổi và invalidate cache khi cần
- **Smart Caching**: Cache dựa trên hash của configuration và file modification time

### Quản lý Cache
```bash
# Xem thông tin cache
python main.py --cache-info

# Xóa cache
python main.py --clear-cache all        # Xóa tất cả
python main.py --clear-cache model      # Chỉ xóa model cache
python main.py --clear-cache dataset    # Chỉ xóa dataset cache

# Buộc tải lại (bỏ qua cache)
python main.py --force-reload
```

### Lợi ích
- **Tăng tốc**: Giảm thời gian setup từ vài phút xuống vài giây
- **Tiết kiệm băng thông**: Không cần tải lại model khi đã có cache
- **Hiệu quả**: Tự động detect thay đổi dataset và config

## Dataset Format

Dataset sử dụng format JSONL với cấu trúc messages:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Bạn là Trợ lý Sale Marketing..."
    },
    {
      "role": "user",
      "content": "Câu hỏi của khách hàng"
    },
    {
      "role": "assistant",
      "content": "Phản hồi của AI"
    }
  ]
}
```

## 🎯 Personas có sẵn

1. **sale_marketing**: Trợ lý Sale Marketing chuyên phân tích và đề xuất chiến lược
2. **nhotin**: Lập trình viên đam mê công nghệ
3. **thang_nguyen**: Lập trình viên tại ngân hàng SHB

## 📝 Logs và Monitoring

- Training logs: `outputs/`
- Model checkpoints: `models/`
- Application logs: `logs/`

## 🔧 Troubleshooting

### Lỗi thường gặp:

1. **CUDA out of memory**:
   - Giảm `per_device_train_batch_size` trong config
   - Tăng `gradient_accumulation_steps`

2. **Model not found**:
   - Chạy training trước: `python main.py`
   - Kiểm tra đường dẫn model

3. **Dataset not found**:
   - Đảm bảo file `data/sale_marketing_finetune_dataset.jsonl` tồn tại
   - Kiểm tra format dataset

4. **Dependencies error**:
   - Kích hoạt đúng conda environment: `conda activate gptoss`
   - Cài đặt lại dependencies

## Workflow hoàn chỉnh

1. **Chuẩn bị**:
   ```bash
   conda activate gptoss
   # Cài đặt dependencies như hướng dẫn trên
   ```

2. **Training**:
   ```bash
   python main.py
   ```

3. **Test model**:
   ```bash
   python inference.py
   ```

4. **Chat interactive**:
   ```
   👤 You: Bên em dùng thử CRM 7 ngày rồi nhưng team chưa quen, giá gói Standard là bao nhiêu?
   Assistant: Ý định: Hỏi giá/chi phí, Yêu cầu dùng thử/demo
   Nỗi lo: Team chưa quen sử dụng
   Phản hồi gợi ý: Cung cấp khung giá rõ ràng...
   ```

## 📈 Performance Tips

- Sử dụng GPU có ít nhất 8GB VRAM
- Tăng `gradient_accumulation_steps` nếu thiếu memory
- Giảm `max_seq_length` nếu cần
- Sử dụng `load_in_4bit=True` để tiết kiệm memory

## 🤝 Contributing

Mọi đóng góp đều được chào đón! Hãy tạo issue hoặc pull request.

## 📄 License

Dự án này sử dụng cho mục đích học tập và nghiên cứu.

---

**Chúc bạn training thành công! 🎉**