import os
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

# Đường dẫn tới thư mục chứa mô hình finetuned
model_path = "/home/nhotin/work/LegalBizAI_project/finetune/embedding_models/baai_bge_m3/output/bge_m3/best_model"
base_model_name = "BAAI/bge-m3"

# Kiểm tra nếu mô hình đã tồn tại trong thư mục cục bộ
if not os.path.exists(model_path):
    print(f"Mô hình finetuned chưa tồn tại tại {model_path}. Đang tải model gốc BAAI/bge-m3...")
    model = SentenceTransformer(base_model_name)
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
                
                # Load base model
                base_model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
                tokenizer = AutoTokenizer.from_pretrained(model_path)  # Load tokenizer từ best_model
                
                # Load và merge LoRA adapter
                model_with_adapter = PeftModel.from_pretrained(base_model, model_path)
                merged_model = model_with_adapter.merge_and_unload()
                
                # Save merged model
                print(f"Đang save merged model vào {merged_model_dir}...")
                merged_model.save_pretrained(merged_model_dir)
                tokenizer.save_pretrained(merged_model_dir)
                
                # Tạo modules.json cho SentenceTransformer (cấu trúc đúng)
                import json
                # SentenceTransformer modules.json format
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
            
            # Load với SentenceTransformer
            model = SentenceTransformer(merged_model_dir)
            print("✓ Đã load model với LoRA adapter thành công!")
            
        except Exception as e:
            print(f"⚠ Lỗi khi load model với adapter: {e}")
            import traceback
            traceback.print_exc()
            print("Fallback: Sử dụng model gốc BAAI/bge-m3...")
            model = SentenceTransformer(base_model_name)
    else:
        # Model không có LoRA, load trực tiếp
        print(f"Đang tải mô hình finetuned từ {model_path}...")
        try:
            model = SentenceTransformer(model_path)
        except Exception as e:
            print(f"⚠ Lỗi khi load model từ {model_path}: {e}")
            print("Fallback: Sử dụng model gốc BAAI/bge-m3...")
            model = SentenceTransformer(base_model_name)

# Kiểm tra mô hình
print(model)

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

faiss_index = faiss.read_index(index_path)


def tokenizer(text):
    text = re.sub(r"[^\w\s%]", "", ud.normalize("NFC", text))
    words = text.split()
    for idx in range(len(words)):
        if words[idx] in bizlaw_short_dict.keys():
            words[idx] = bizlaw_short_dict[words[idx]]
    temp = " ".join(words).lower()
    return temp

def get_embedding(text):
    return model.encode(text)

def load_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)
    
def get_question_embedding(question):
    return get_embedding(question)

def retrieve(ques, topk=3):
    # Tokenize câu hỏi trước
    tokenized_ques = tokenizer(ques)
    # Encode câu hỏi đã tokenize
    ques_embedding = model.encode(tokenized_ques)
    _, I = faiss_index.search(np.array([ques_embedding]), k=topk)
    return I[0].tolist()
