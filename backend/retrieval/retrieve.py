import os
# Đảm bảo chạy trên CPU - KHÔNG dùng GPU (phải set TRƯỚC khi import torch)
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Vô hiệu hóa GPU

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

# Đường dẫn tới thư mục chứa mô hình fine-tuned
model_base_path = "/home/nhotin/work/LegalBizAI_project/finetune/embedding_models/baai_bge_m3"
adapter_path = os.path.join(model_base_path, "output/bge_m3/final_model")
base_model_name = "BAAI/bge-m3"

device = "cpu"  # Chỉ dùng CPU

# Model sẽ được load lazy (chỉ khi cần)
_model = None

def _load_model():
    """Lazy load model - chỉ load khi hàm này được gọi lần đầu"""
    global _model
    
    if _model is not None:
        return _model
    
    print(f"⚠️  Đang load model embedding trên {device} (lần đầu tiên)...")
    
    # Kiểm tra nếu adapter tồn tại
    if os.path.exists(adapter_path) and os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        print(f"Đang load model từ adapter: {adapter_path}")
        try:
            # Load base model với device_map="cpu" để đảm bảo không dùng GPU
            base_model = AutoModel.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # Sử dụng float32 cho CPU
                device_map="cpu"  # Force CPU
            )
            base_model = base_model.to(device)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            
            # Load adapter
            model_with_adapter = PeftModel.from_pretrained(base_model, adapter_path)
            model_with_adapter = model_with_adapter.to(device)
            model_with_adapter.eval()
            
            # Wrap trong SentenceTransformer để sử dụng như bình thường
            # Tạo một wrapper class để encode
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
                    
                    # Batch processing for efficiency
                    inputs = self.tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_seq_length,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        # Mean pooling (same as BGE-M3)
                        attention_mask = inputs["attention_mask"]
                        token_embeddings = outputs.last_hidden_state
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                        # Normalize embeddings
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                        embeddings = embeddings.cpu().numpy()
                    
                    # Return single embedding if input was string, otherwise return list
                    if is_single_text:
                        return embeddings[0]
                    return embeddings
            
            _model = AdapterModelWrapper(model_with_adapter, tokenizer, device)
            print("✅ Model đã được load với adapter fine-tuned (CPU only)")
        except Exception as e:
            print(f"❌ Lỗi khi load adapter, sẽ load base model: {e}")
            # Fallback: load base model
            _model = SentenceTransformer(base_model_name, device=device)
else:
        print(f"Không tìm thấy adapter, đang load base model từ {base_model_name}")
        _model = SentenceTransformer(base_model_name, device=device)

    return _model

# Các hàm và biến khác
import json
import re
import unicodedata as ud
import faiss
import numpy as np
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain._api.module_import")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.utils.generic")

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
stop_words_vn = set(
    [
        "và",
        "của",
        "là",
        "các",
        "trong",
        "với",
        "cho",
        "để",
        "những",
        "khi",
        "thì",
        "này",
        "làm",
        "từ",
        "đã",
        "sẽ",
        "rằng",
        "mà",
        "như",
        "lại",
        "ra",
        "sau",
        "cũng",
        "vậy",
        "nếu",
        "đến",
        "thế",
        "biết",
        "theo",
        "đâu",
        "đó",
        "trước",
        "vừa",
        "rồi",
        "trên",
        "dưới",
        "ngoài",
        "gì",
        "còn",
        "nữa",
        "nào",
        "hết",
        "ai",
        "ấy",
        "lúc",
        "ở",
        "đi",
        "về",
        "ngay",
        "luôn",
        "đang",
        "thì",
        "đây",
        "kia",
        "ấy",
        "điều",
        "việc",
        "vì",
        "giữa",
        "qua",
        "vẫn",
        "chỉ",
        "nói",
        "thật",
        "hơn",
        "vậy",
        "hay",
        "lại",
        "ngày",
        "giờ",
        "tại",
        "bởi",
        "sao",
        "trước",
        "sau",
        "đó",
        "mà",
        "về",
        "đến",
        "thì",
        "được",
        "thế",
        "còn",
        "đến",
        "cũng",
        "này",
        "đấy",
        "một",
        "vì",
        "những",
        "thì",
        "vậy",
        "thế",
        "đây",
        "vẫn",
        "lại",
        "thì",
        "còn",
        "đó",
        "này",
        "ở",
        "trong",
        "làm",
        "khi",
        "vậy",
        "này",
        "đó",
        "ở",
        "được",
        "làm",
        "để",
        "khi",
        "với",
        "về",
        "đi",
        "cho",
        "về",
        "đã",
        "với",
        "như",
        "đi",
        "này",
        "như",
        "được",
        "cho",
        "thì",
        "làm",
        "ở",
        "như",
        "điều",
        "khi",
        "với",
        "trong",
    ]
)

index_path = "./data/faiss_index"

# Index sẽ được load lazy (chỉ khi cần)
_faiss_index = None

def _load_index():
    """Lazy load FAISS index - chỉ load khi hàm này được gọi lần đầu"""
    global _faiss_index
    
    if _faiss_index is not None:
        return _faiss_index
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index không tồn tại tại {index_path}. "
            f"Hãy chạy script build_index.py để tạo index trước."
        )
    
    print(f"📖 Đang load FAISS index từ {index_path}...")
    _faiss_index = faiss.read_index(index_path)
    print(f"✅ Đã load FAISS index với {_faiss_index.ntotal} vectors")
    
    return _faiss_index

def tokenizer(text):
    text = re.sub(r"[^\w\s%]", "", ud.normalize("NFC", text))
    words = text.split()
    for idx in range(len(words)):
        if words[idx] in bizlaw_short_dict.keys():
            words[idx] = bizlaw_short_dict[words[idx]]
    temp = " ".join(words).lower()
    return temp

def get_embedding(text):
    model = _load_model()  # Lazy load khi cần
    return model.encode(text)

def load_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)
    
def get_question_embedding(question):
    return get_embedding(question)

def retrieve(ques, topk=3):
    model = _load_model()  # Lazy load khi cần
    faiss_index = _load_index()  # Lazy load khi cần
    # Tokenize câu hỏi trước
    tokenized_ques = tokenizer(ques)
    # Encode câu hỏi đã tokenize
    ques_embedding = model.encode(tokenized_ques)
    _, I = faiss_index.search(np.array([ques_embedding]), k=topk)
    return I[0].tolist()
