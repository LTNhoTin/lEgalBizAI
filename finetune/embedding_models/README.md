# Embedding Models Fine-tuning for LegalBizAI

Hệ thống fine-tuning cho 3 model embedding chuyên biệt cho domain pháp lý Việt Nam.

## Tổng quan

Dự án này triển khai fine-tuning cho 3 model embedding state-of-the-art:

1. **Google Gemma Embeddings** - Model embedding mới nhất từ Google
2. **BAAI bge-m3** - Model embedding đa ngôn ngữ hiệu suất cao
3. **all-MiniLM-L6-v2** - Model SBERT compact và hiệu quả

## 🎯 Mục tiêu đánh giá

| Metric                | Định nghĩa                                                                             | Mục tiêu (Target) |
| --------------------- | -------------------------------------------------------------------------------------- | ----------------- |
| **Recall@k**          | % truy vấn mà đoạn "gold passage" nằm trong top-k kết quả truy xuất                    | ≥ **85%**         |
| **MRR@k**             | Trung bình nghịch đảo của vị trí đoạn liên quan đầu tiên (Mean Reciprocal Rank)        | ≥ **0.8**         |
| **nDCG@k**            | Đánh giá chất lượng xếp hạng có xét đến vị trí (Normalized Discounted Cumulative Gain) | ≥ **0.8**         |
| **Context Precision** | Tỷ lệ phần trăm các đoạn được truy xuất thật sự liên quan                              | ≥ **85%**         |

## 📁 Cấu trúc thư mục

```
embedding_models/
├── README.md                           # Tài liệu này
├── WANDB_LOGGING_GUIDE.md              # Hướng dẫn chi tiết W&B logging
├── requirements.txt                    # Dependencies chung
├── data_loader.py                      # Xử lý dữ liệu
├── evaluation_utils.py                 # Utilities đánh giá
├── train_all_models.py                 # Script training tất cả models
├── evaluate_all_models.py              # Script đánh giá tất cả models
├── test_wandb_logging.py               # Test W&B logging functionality
│
├── google_gemma_embeddings/            # Google Gemma Embeddings
│   ├── config/
│   │   └── config.py                   # Cấu hình training
│   ├── src/
│   │   └── trainer.py                  # Logic training (với W&B logging)
│   ├── train.py                        # Script training chính
│   ├── models/                         # Models đã train
│   ├── logs/                           # Training logs
│   ├── wandb/                          # W&B local cache
│   └── evaluation/                     # Kết quả đánh giá
│
├── baai_bge_m3/                        # BAAI bge-m3
│   ├── config/
│   │   └── config.py
│   ├── src/
│   │   └── trainer.py                  # Logic training (với W&B logging)
│   ├── train.py
│   ├── models/
│   ├── logs/
│   ├── wandb/                          # W&B local cache
│   └── evaluation/
│
└── all_minilm_l6_v2/                   # all-MiniLM-L6-v2
    ├── config/
    │   └── config.py
    ├── src/
    │   └── trainer.py                  # Logic training (với W&B logging)
    ├── train.py
    ├── models/
    ├── logs/
    ├── wandb/                          # W&B local cache
    └── evaluation/
```

## Cài đặt và Thiết lập

### 1. Cài đặt Dependencies

```bash
cd /home/nhotin/work/LegalBizAI_project/finetune/embedding_models
pip install -r requirements.txt
```

### 2. Kiểm tra Dependencies

```bash
python train_all_models.py --check_deps
```

## 🏃‍♂️ Cách sử dụng

### Training một model cụ thể

#### Google Gemma Embeddings
```bash
cd google_gemma_embeddings
python train.py --batch_size 4 --num_epochs 3 --use_wandb
```

#### BAAI bge-m3
```bash
cd baai_bge_m3
python train.py --batch_size 8 --num_epochs 3 --use_wandb
```

#### all-MiniLM-L6-v2
```bash
cd all_minilm_l6_v2
python train.py --batch_size 16 --num_epochs 4 --use_wandb
```

### Training tất cả models

```bash
# Training tất cả models với cấu hình mặc định
python train_all_models.py

# Training với custom parameters
python train_all_models.py --batch_size 8 --num_epochs 5 --use_wandb

# Training một model cụ thể
python train_all_models.py --model gemma --batch_size 4
python train_all_models.py --model bge --batch_size 8
python train_all_models.py --model minilm --batch_size 16
```

### Đánh giá models

```bash
# Đánh giá tất cả models
python evaluate_all_models.py --output evaluation_results.txt

# Đánh giá với custom k values
python evaluate_all_models.py --k_values 1 3 5 10 20 50
```

## ⚙️ Cấu hình Training

### Google Gemma Embeddings
- **Batch size**: 4 (do model lớn)
- **Learning rate**: 1e-5
- **Epochs**: 3
- **LoRA**: Enabled (r=16, alpha=32)
- **Loss**: Contrastive Loss

### BAAI bge-m3
- **Batch size**: 8
- **Learning rate**: 1e-5
- **Epochs**: 3
- **LoRA**: Enabled (r=8, alpha=16)
- **Loss**: Multi-representation Loss (Dense + Sparse + ColBERT)

### all-MiniLM-L6-v2
- **Batch size**: 16
- **Learning rate**: 2e-5
- **Epochs**: 4
- **Loss**: Multiple Negatives Ranking Loss

## Monitoring và Logging

### Weights & Biases (W&B)
Hệ thống đã được tích hợp đầy đủ với W&B để theo dõi training process.

#### Thiết lập W&B
```bash
# Đăng nhập W&B (chỉ cần làm 1 lần)
wandb login

# Enable W&B logging khi training
python train.py --use_wandb
```

#### Metrics được track trên W&B

**Training Metrics:**
- `train/loss`: Loss value theo từng step
- `train/learning_rate`: Learning rate schedule
- `train/epoch`: Epoch hiện tại
- `train/global_step`: Tổng số steps đã train
- `train/gradient_norm`: Gradient norm để monitor stability
- `train/batch_size`: Batch size hiện tại

**System Metrics:**
- `system/gpu_memory_gb`: GPU memory usage
- `system/epoch_progress`: Tiến độ epoch (%)

**Model Architecture:**
- `model/total_parameters`: Tổng số parameters
- `model/trainable_parameters`: Số parameters có thể train
- `model/trainable_percentage`: % parameters trainable

**Evaluation Metrics:**
- `eval/recall@k`: Recall tại các k values
- `eval/mrr@k`: Mean Reciprocal Rank
- `eval/ndcg@k`: Normalized DCG
- `eval/context_precision@k`: Context precision
- `eval/num_queries`: Số lượng queries đánh giá
- `eval/num_documents`: Số lượng documents

**Final Metrics:**
- `final/recall@k`: Kết quả cuối cùng
- `final/mrr@k`: MRR cuối cùng
- `final/ndcg@k`: nDCG cuối cùng
- `final/context_precision@k`: Context precision cuối cùng

#### W&B Dashboard Features
- **Real-time monitoring**: Theo dõi training progress
- **Hyperparameter tracking**: Tự động log tất cả config
- **Model comparison**: So sánh performance giữa các models
- **System monitoring**: GPU usage, memory consumption
- **Artifact tracking**: Lưu trữ model checkpoints

#### Xem kết quả trên W&B
```bash
# Mở W&B dashboard
wandb dashboard

# Hoặc truy cập: https://wandb.ai/your-username/embedding-models
```

### Local Logs
Training logs được lưu trong thư mục `logs/` của mỗi model:
- `training.log`: Chi tiết quá trình training
- `evaluation.log`: Kết quả đánh giá
- `wandb/`: W&B local cache và metadata

## 🔍 Đánh giá và Metrics

### Metrics được tính toán
- **Recall@k**: Tỷ lệ truy xuất đúng trong top-k
- **MRR@k**: Mean Reciprocal Rank
- **nDCG@k**: Normalized Discounted Cumulative Gain
- **Context Precision**: Độ chính xác của context

### K values mặc định
`[1, 3, 5, 10, 20]`

### Kết quả đánh giá
Kết quả được lưu trong:
- `evaluation_results.txt`: Báo cáo chi tiết
- `evaluation_results.json`: Dữ liệu JSON để xử lý

## 🛠️ Troubleshooting

### Lỗi thường gặp

#### 1. Out of Memory (OOM)
```bash
# Giảm batch size
python train.py --batch_size 2

# Hoặc disable mixed precision
python train.py --no_fp16
```

#### 2. CUDA không khả dụng
```bash
# Kiểm tra CUDA
python -c \"import torch; print(torch.cuda.is_available())\"

# Training trên CPU (chậm hơn)
python train.py --device cpu
```

#### 3. Dependencies thiếu
```bash
# Cài đặt lại dependencies
pip install -r requirements.txt --force-reinstall
```

#### 4. W&B logging không hoạt động
```bash
# Kiểm tra W&B login
wandb status

# Login lại nếu cần
wandb login

# Test W&B connection
python test_wandb_logging.py

# Disable W&B nếu gặp vấn đề
python train.py --no_wandb
```

### Performance Tuning

#### Tăng tốc training
1. **Sử dụng mixed precision**: `--fp16` (mặc định enabled)
2. **Tăng batch size**: Nếu GPU memory đủ
3. **Gradient accumulation**: Tăng `gradient_accumulation_steps`
4. **Dataloader workers**: Tăng `dataloader_num_workers`

#### Cải thiện chất lượng model
1. **Tăng số epochs**: `--num_epochs`
2. **Điều chỉnh learning rate**: `--learning_rate`
3. **Thêm negative samples**: `--negative_samples`
4. **Tăng sequence length**: `--max_seq_length`

## 📈 Kết quả mong đợi

### Baseline Performance
Dựa trên các nghiên cứu tương tự, kỳ vọng:

| Model | Recall@5 | MRR@10 | nDCG@10 |
|-------|----------|---------|---------|
| Gemma | 82-88% | 0.78-0.85 | 0.76-0.82 |
| BGE-M3 | 85-90% | 0.80-0.87 | 0.78-0.84 |
| MiniLM | 80-85% | 0.75-0.82 | 0.73-0.79 |

### Thời gian training ước tính
- **Gemma**: 2-4 giờ (GPU V100/A100)
- **BGE-M3**: 1-3 giờ
- **MiniLM**: 30-90 phút

## 🔄 Workflow khuyến nghị

1. **Kiểm tra dependencies**:
   ```bash
   python train_all_models.py --check_deps
   ```

2. **Training thử nghiệm** (ít epochs):
   ```bash
   python train_all_models.py --num_epochs 1
   ```

3. **Training đầy đủ**:
   ```bash
   python train_all_models.py --use_wandb
   ```

4. **Đánh giá kết quả**:
   ```bash
   python evaluate_all_models.py
   ```

5. **Phân tích và so sánh**:
   - Xem file `evaluation_results.txt`
   - Kiểm tra W&B dashboard
   - So sánh với target metrics

## 📞 Hỗ trợ

Nếu gặp vấn đề:
1. Kiểm tra logs trong thư mục `logs/`
2. Xem file `training.log` và `evaluation.log`
3. Kiểm tra GPU memory: `nvidia-smi`
4. Verify data format trong `/finetune/data_finetunning/`

## 📝 Notes

- **Data format**: QA pairs trong JSON với fields: `question`, `answer`, `references`
- **GPU requirements**: Khuyến nghị ≥8GB VRAM cho training hiệu quả
- **Storage**: Mỗi model cần ~2-5GB storage cho checkpoints
- **Evaluation**: Sử dụng 20% cuối của dataset làm test set

---

**Happy Fine-tuning! 🚀**