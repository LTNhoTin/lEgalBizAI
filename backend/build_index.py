"""
Script để build FAISS index từ dữ liệu chunks
"""
import json
import faiss
import numpy as np
import re
import unicodedata as ud
from tqdm import tqdm

# Copy tokenizer function từ retrieve.py để tránh import toàn bộ module
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
    text = re.sub(r"[^\w\s%]", "", ud.normalize("NFC", text))
    words = text.split()
    for idx in range(len(words)):
        if words[idx] in bizlaw_short_dict.keys():
            words[idx] = bizlaw_short_dict[words[idx]]
    temp = " ".join(words).lower()
    return temp

def build_faiss_index(chunks_file="./data/all_chunks_final.json", 
                     index_path="./data/faiss_index",
                     batch_size=32):
    """
    Build FAISS index từ file chunks JSON
    
    Args:
        chunks_file: Đường dẫn tới file JSON chứa chunks
        index_path: Đường dẫn để lưu FAISS index
        batch_size: Kích thước batch để encode embeddings
    """
    print(f"📖 Đang đọc dữ liệu từ {chunks_file}...")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"✅ Đã load {len(chunks)} chunks")
    
    # Load model - chỉ dùng adapter, không fallback về SentenceTransformer
    print("🤖 Đang load model embedding từ adapter...")
    # Import các module cần thiết
    import os
    import torch
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel
    
    model_base_path = "/home/nhotin/work/LegalBizAI_project/finetune/embedding_models/baai_bge_m3"
    adapter_path = os.path.join(model_base_path, "output/bge_m3/final_model")
    base_model_name = "BAAI/bge-m3"
    device = "cpu"
    
    if os.path.exists(adapter_path) and os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        print(f"✅ Đang load model từ adapter: {adapter_path}")
        base_model = AutoModel.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            dtype=torch.float32,
            device_map="cpu",
            use_safetensors=True
        )
        base_model = base_model.to(device)
        tokenizer_model = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        model_with_adapter = PeftModel.from_pretrained(base_model, adapter_path)
        model_with_adapter = model_with_adapter.to(device)
        model_with_adapter.eval()
        
        # Wrap trong class để encode
        class AdapterModelWrapper:
            def __init__(self, model, tokenizer, device):
                self.model = model
                self.tokenizer = tokenizer
                self.device = device
                self.max_seq_length = 512
                
            def encode(self, texts, **kwargs):
                is_single_text = isinstance(texts, str)
                if is_single_text:
                    texts = [texts]
                
                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    attention_mask = inputs["attention_mask"]
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    embeddings = embeddings.cpu().numpy()
                
                if is_single_text:
                    return embeddings[0]
                return embeddings
        
        model = AdapterModelWrapper(model_with_adapter, tokenizer_model, device)
        print("✅ Model đã được load với adapter fine-tuned")
    else:
        raise FileNotFoundError(f"Adapter không tồn tại tại {adapter_path}")
    
    # Lấy embedding dimension từ model (thử encode một sample)
    sample_text = chunks[0]["passage"] if chunks else "sample"
    sample_embedding = model.encode(sample_text)
    dimension = len(sample_embedding)
    print(f"📐 Embedding dimension: {dimension}")
    
    # Tạo FAISS index
    # Sử dụng IndexFlatIP (Inner Product) cho cosine similarity (vì embeddings đã được normalize)
    # hoặc IndexFlatL2 cho L2 distance
    index = faiss.IndexFlatL2(dimension)
    
    # Encode tất cả passages và thêm vào index
    print(f"🔄 Đang encode {len(chunks)} passages...")
    embeddings_list = []
    
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch_chunks = chunks[i:i+batch_size]
        batch_texts = []
        
        for chunk in batch_chunks:
            # Tokenize và encode passage
            tokenized_text = tokenizer(chunk["passage"])
            batch_texts.append(tokenized_text)
        
        # Encode batch
        batch_embeddings = model.encode(batch_texts, batch_size=batch_size, show_progress_bar=False)
        embeddings_list.append(batch_embeddings)
    
    # Convert sang numpy array và normalize
    all_embeddings = np.vstack(embeddings_list).astype('float32')
    
    # Normalize embeddings cho cosine similarity
    faiss.normalize_L2(all_embeddings)
    
    # Thêm vào index
    print("📊 Đang thêm embeddings vào FAISS index...")
    index.add(all_embeddings)
    
    print(f"✅ Index đã được tạo với {index.ntotal} vectors")
    
    # Lưu index
    print(f"💾 Đang lưu index vào {index_path}...")
    faiss.write_index(index, index_path)
    print(f"✅ Đã lưu index thành công!")
    
    return index

if __name__ == "__main__":
    import sys
    
    chunks_file = sys.argv[1] if len(sys.argv) > 1 else "./data/all_chunks_final.json"
    index_path = sys.argv[2] if len(sys.argv) > 2 else "./data/faiss_index"
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    
    build_faiss_index(chunks_file, index_path, batch_size)

