import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

# Đảm bảo chạy trên CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(4)  # Giới hạn số threads để tránh quá tải CPU

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
                        torch_dtype=torch.float32,  # Sử dụng float32 cho CPU
                        device_map="cpu",
                        use_safetensors=True  # Ưu tiên safetensors để tránh lỗi PyTorch version
                    )
                except Exception as e:
                    # Fallback nếu không có safetensors
                    print(f"⚠ Không thể load với safetensors: {e}. Thử load không safetensors...")
                    base_model = AutoModel.from_pretrained(
                        base_model_name, 
                        trust_remote_code=True,
                        torch_dtype=torch.float32,
                        device_map="cpu"
                    )
                tokenizer = AutoTokenizer.from_pretrained(model_path)  # Load tokenizer từ best_model
                
                # Load và merge LoRA adapter
                model_with_adapter = PeftModel.from_pretrained(base_model, model_path)
                merged_model = model_with_adapter.merge_and_unload()
                
                # Save merged model với safetensors
                print(f"Đang save merged model vào {merged_model_dir}...")
                merged_model.save_pretrained(merged_model_dir, safe_serialization=True)
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
                # Kiểm tra xem có safetensors files không
                import glob
                safetensors_files = glob.glob(os.path.join(merged_model_dir, "*.safetensors"))
                if not safetensors_files:
                    print("⚠ Merged model không có safetensors files. Có thể gặp lỗi PyTorch version.")
                    print("   Đề xuất: Xóa merged_model và merge lại để tạo safetensors files.")
            
            # Load với SentenceTransformer trên CPU
            print("⚠ Đảm bảo chạy trên CPU...")
            try:
                # Thử load bằng SentenceTransformer trước
                model = SentenceTransformer(merged_model_dir, device="cpu")
                print("✓ Đã load model với LoRA adapter thành công!")
            except (ValueError, AttributeError, Exception) as e:
                # Nếu không load được bằng SentenceTransformer, load trực tiếp bằng AutoModel
                print(f"⚠ Không thể load bằng SentenceTransformer: {e}")
                print("Đang load trực tiếp bằng AutoModel...")
                try:
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
                                
                                # Trả về 1D array nếu input là string đơn
                                if is_single:
                                    return result[0]
                                return result
                        
                        def to(self, device):
                            self.model = self.model.to(device)
                            return self
                    
                    model = ModelWrapper(merged_model, merged_tokenizer)
                    print("✓ Đã load model với LoRA adapter thành công (AutoModel wrapper)!")
                except Exception as e2:
                    print(f"⚠ Lỗi khi load merged model: {e2}")
                    raise e
            
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

# Kiểm tra mô hình và đảm bảo trên CPU
print(model)
if hasattr(model, '_modules'):
    for module_name, module in model._modules.items():
        if hasattr(module, 'to'):
            module = module.to('cpu')
print("✓ Model đã được đảm bảo chạy trên CPU")

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

# Lấy đường dẫn tuyệt đối của thư mục backend
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
index_path = os.path.join(BASE_DIR, "data", "faiss_index")

try:
    if os.path.exists(index_path):
        faiss_index = faiss.read_index(index_path)
        print(f"✓ Đã load faiss_index thành công từ {index_path}")
    else:
        faiss_index = None
        import warnings
        warnings.warn(f"⚠ File faiss_index không tồn tại tại {index_path}. Retrieval sẽ không hoạt động.")
except (RuntimeError, FileNotFoundError, Exception) as e:
    faiss_index = None
    import warnings
    warnings.warn(f"⚠ Không thể load faiss_index từ {index_path}: {e}. Retrieval sẽ không hoạt động.")


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
    if faiss_index is None:
        import warnings
        warnings.warn("⚠ faiss_index chưa được load. Trả về empty list. Vui lòng kiểm tra file faiss_index.")
        # Trả về empty list thay vì raise exception để không crash app
        return []
    # Tokenize câu hỏi trước
    tokenized_ques = tokenizer(ques)
    # Encode câu hỏi đã tokenize
    ques_embedding = model.encode(tokenized_ques)
    _, I = faiss_index.search(np.array([ques_embedding]), k=topk)
    return I[0].tolist()
