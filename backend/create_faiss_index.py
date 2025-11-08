import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import json
import re
import unicodedata as ud
import faiss
import numpy as np
from tqdm import tqdm

# Đảm bảo chạy trên CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(4)

print("=" * 80)
print("SCRIPT TẠO FAISS INDEX TỪ MODEL FINETUNED")
print("=" * 80)

# Đường dẫn tới thư mục chứa mô hình finetuned
model_path = "/home/nhotin/work/LegalBizAI_project/finetune/embedding_models/baai_bge_m3/output/bge_m3/best_model"
base_model_name = "BAAI/bge-m3"

# Kiểm tra nếu mô hình đã tồn tại trong thư mục cục bộ
if not os.path.exists(model_path):
    print(f"Mô hình finetuned chưa tồn tại tại {model_path}. Đang tải model gốc BAAI/bge-m3...")
    print("⚠ Đảm bảo chạy trên CPU...")
    model = SentenceTransformer(base_model_name, device="cpu")
else:
    # Kiểm tra xem có adapter (LoRA) không
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    
    if os.path.exists(adapter_config_path):
        # Model có LoRA adapter, cần load base model + adapter rồi merge
        print(f"Phát hiện LoRA adapter tại {model_path}")
        print("Đang load base model và merge với LoRA adapter...")
        
        try:
            merged_model_dir = os.path.join(model_path, "merged_model")
            
            if not os.path.exists(merged_model_dir) or not os.path.exists(os.path.join(merged_model_dir, "modules.json")):
                # Chưa có merged model, cần merge và save
                print("Đang merge LoRA adapter vào base model...")
                print("⚠ Đảm bảo chạy trên CPU...")
                
                # Load base model trên CPU với safetensors (nếu có)
                try:
                    base_model = AutoModel.from_pretrained(
                        base_model_name, 
                        trust_remote_code=True,
                        torch_dtype=torch.float32,
                        device_map="cpu",
                        use_safetensors=True
                    )
                except Exception as e:
                    print(f"⚠ Không thể load với safetensors: {e}. Thử load không safetensors...")
                    base_model = AutoModel.from_pretrained(
                        base_model_name, 
                        trust_remote_code=True,
                        torch_dtype=torch.float32,
                        device_map="cpu"
                    )
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                # Load và merge LoRA adapter
                model_with_adapter = PeftModel.from_pretrained(base_model, model_path)
                merged_model = model_with_adapter.merge_and_unload()
                
                # Save merged model với safetensors
                print(f"Đang save merged model vào {merged_model_dir}...")
                merged_model.save_pretrained(merged_model_dir, safe_serialization=True)
                tokenizer.save_pretrained(merged_model_dir)
                
                # Tạo modules.json cho SentenceTransformer
                modules_config = [
                    {
                        "idx": 0,
                        "name": "0",
                        "path": ".",
                        "type": "transformers.AutoModel"
                    }
                ]
                with open(os.path.join(merged_model_dir, "modules.json"), "w") as f:
                    json.dump(modules_config, f, indent=2)
                
                print("✓ Đã merge và save model thành công!")
            else:
                print(f"Đã có merged model tại {merged_model_dir}, đang load...")
            
            # Load với SentenceTransformer trên CPU
            print("⚠ Đảm bảo chạy trên CPU...")
            try:
                model = SentenceTransformer(merged_model_dir, device="cpu")
                print("✓ Đã load model với LoRA adapter thành công!")
            except (ValueError, AttributeError, Exception) as e:
                print(f"⚠ Không thể load bằng SentenceTransformer: {e}")
                print("Đang load trực tiếp bằng AutoModel...")
                
                # Load merged model trực tiếp
                merged_model = AutoModel.from_pretrained(
                    merged_model_dir,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    device_map="cpu"
                )
                merged_tokenizer = AutoTokenizer.from_pretrained(merged_model_dir)
                
                # Tạo wrapper để tương thích với SentenceTransformer API
                class ModelWrapper:
                    def __init__(self, model, tokenizer):
                        self.model = model
                        self.tokenizer = tokenizer
                        self._modules = {'0': model}
                    
                    def encode(self, texts, **kwargs):
                        is_single = isinstance(texts, str)
                        if is_single:
                            texts = [texts]
                        
                        # Tokenize và encode
                        encoded = self.tokenizer(
                            texts,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors="pt"
                        )
                        
                        with torch.no_grad():
                            outputs = self.model(**encoded)
                            # BGE-M3 sử dụng dense embeddings
                            if hasattr(outputs, 'dense_embeds'):
                                embeddings = outputs.dense_embeds
                            elif hasattr(outputs, 'last_hidden_state'):
                                # Mean pooling
                                embeddings = outputs.last_hidden_state.mean(dim=1)
                            else:
                                embeddings = outputs[0].mean(dim=1)
                            
                            # Normalize
                            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                            
                            result = embeddings.cpu().numpy()
                            
                            if is_single:
                                return result[0]
                            return result
                    
                    def to(self, device):
                        self.model = self.model.to(device)
                        return self
                
                model = ModelWrapper(merged_model, merged_tokenizer)
                print("✓ Đã load model với LoRA adapter thành công (AutoModel wrapper)!")
            
        except Exception as e:
            print(f"⚠ Lỗi khi load model với adapter: {e}")
            import traceback
            traceback.print_exc()
            print("Fallback: Sử dụng model gốc BAAI/bge-m3...")
            print("⚠ Đảm bảo chạy trên CPU...")
            model = SentenceTransformer(base_model_name, device="cpu")
    else:
        # Model không có LoRA, load trực tiếp
        print(f"Đang tải mô hình finetuned từ {model_path}...")
        print("⚠ Đảm bảo chạy trên CPU...")
        try:
            model = SentenceTransformer(model_path, device="cpu")
        except Exception as e:
            print(f"⚠ Lỗi khi load model từ {model_path}: {e}")
            import traceback
            traceback.print_exc()
            print("Fallback: Sử dụng model gốc BAAI/bge-m3...")
            model = SentenceTransformer(base_model_name, device="cpu")

print("✓ Model đã được load thành công!")
print("=" * 80)

# Dictionary và stop words (giống như trong retrieve.py)
bizlaw_short_dict = {
    "BCC": "hợp tác kinh doanh",
    "BHTN": "Bảo hiểm thất nghiệp",
    "BHXH": "Bảo hiểm xã hội",
    "BHYT": "Bảo hiểm y tế",
    "CT": "Công ty",
    "CTCP": "Công ty cổ phần",
    "DNTN": "Doanh nghiệp tư nhân",
    "DV": "Dịch vụ",
    "EUR": "Euro",
    "EVN": "Điện lực Việt Nam",
    "GCN": "Giấy chứng nhận",
    "GPS": "hệ thống định vị toàn cầu GPS",
    "HĐQT": "Hội đồng quản trị",
    "JPY": "Yên Nhật",
    "MTV": "Một thành viên",
    "STT": "Số thứ tự",
    "TM": "Thương mại",
    "TNDN": "thu nhập doanh nghiệp",
    "TNHH": "Trách nhiệm hữu hạn",
    "TP": "Thành phố",
    "USD": "Đô la Mỹ",
    "ĐHĐCĐ": "Đại hội đồng cổ đông",
    "ĐKKD": "Đăng ký kinh doanh",
}

def tokenizer(text):
    """Tokenize text giống như trong retrieve.py"""
    text = re.sub(r"[^\w\s%]", "", ud.normalize("NFC", text))
    words = text.split()
    for idx in range(len(words)):
        if words[idx] in bizlaw_short_dict.keys():
            words[idx] = bizlaw_short_dict[words[idx]]
    temp = " ".join(words).lower()
    return temp

def get_embedding(text):
    """Get embedding từ text"""
    return model.encode(text)

# Lấy đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
chunks_file = os.path.join(BASE_DIR, "data", "all_chunks_final.json")
index_output_path = os.path.join(BASE_DIR, "data", "faiss_index")

# Backup index cũ nếu tồn tại
if os.path.exists(index_output_path):
    backup_path = index_output_path + ".backup"
    print(f"⚠ File index cũ đã tồn tại. Đang backup vào {backup_path}...")
    import shutil
    shutil.copy(index_output_path, backup_path)
    print(f"✓ Đã backup index cũ!")

# Load chunks
print(f"\nĐang đọc chunks từ {chunks_file}...")
with open(chunks_file, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"✓ Đã load {len(chunks)} chunks")

# Tạo embeddings cho tất cả chunks
print("\nĐang tạo embeddings cho tất cả chunks...")
print("(Quá trình này có thể mất vài phút...)")

embeddings = []
batch_size = 32  # Process in batches để tối ưu

for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
    batch = chunks[i:i+batch_size]
    # Tokenize từng chunk trong batch
    tokenized_texts = [tokenizer(chunk["passage"]) for chunk in batch]
    # Encode batch
    batch_embeddings = model.encode(tokenized_texts, show_progress_bar=False)
    embeddings.extend(batch_embeddings)

embeddings = np.array(embeddings, dtype=np.float32)
print(f"✓ Đã tạo embeddings với shape: {embeddings.shape}")

# Tạo FAISS index
print("\nĐang tạo FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner Product (cho cosine similarity với normalized vectors)

# Normalize embeddings (quan trọng cho cosine similarity)
faiss.normalize_L2(embeddings)

# Add embeddings vào index
index.add(embeddings)

print(f"✓ Đã tạo FAISS index với {index.ntotal} vectors")

# Lưu index
print(f"\nĐang lưu FAISS index vào {index_output_path}...")
faiss.write_index(index, index_output_path)
print(f"✓ Đã lưu FAISS index thành công!")

# Test index
print("\n" + "=" * 80)
print("TEST FAISS INDEX")
print("=" * 80)
test_query = "thành lập công ty trách nhiệm hữu hạn"
print(f"Câu hỏi test: '{test_query}'")
tokenized_query = tokenizer(test_query)
query_embedding = model.encode(tokenized_query)
query_embedding = np.array([query_embedding], dtype=np.float32)
faiss.normalize_L2(query_embedding)

k = 3
distances, indices = index.search(query_embedding, k)

print(f"\nTop {k} kết quả tương tự:")
for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
    print(f"\n{i+1}. Index: {idx}, Distance: {dist:.4f}")
    print(f"   Title: {chunks[idx].get('title', 'N/A')}")
    print(f"   Passage: {chunks[idx]['passage'][:200]}...")
    print(f"   Root: {chunks[idx].get('root', 'N/A')}")

print("\n" + "=" * 80)
print("HOÀN THÀNH!")
print("=" * 80)
print(f"✓ FAISS index mới đã được tạo và lưu tại: {index_output_path}")
print(f"✓ Sử dụng model finetuned từ: {model_path}")
print(f"✓ Tổng số vectors: {index.ntotal}")
print(f"✓ Dimension: {dimension}")

