#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI Server cho GPT-OSS 20B Finetuned Model
Chạy trên port 1143 với 4-bit quantization để phù hợp VRAM 16GB
"""

import os
import torch
import gc
import shutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
import logging

# Import unsloth
from unsloth import FastLanguageModel
from transformers import TextStreamer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================
MODEL_BASE_PATH = "/home/nhotin/work/LegalBizAI_project/finetune/gpt_oss_20b_finetune"
MAX_SEQ_LENGTH = 1024
PORT = 1143

# ============================================
# MODEL PATH SELECTION
# ============================================
# Có thể chọn model path bằng biến môi trường MODEL_PATH
# Nếu không set, sẽ dùng giá trị mặc định bên dưới

# Các thư mục có sẵn:
# - finetuned_model/ (adapter model)
# - outputs/checkpoint-500/
# - outputs/checkpoint-1000/
# - outputs/checkpoint-1500/
# - outputs/checkpoint-2000/
# - outputs/checkpoint-2500/
# - outputs/checkpoint-3000/
# - outputs/checkpoint-3175/ (checkpoint cuối cùng)

# CHỌN MODEL PATH: Uncomment một trong các dòng dưới để chọn thư mục muốn dùng
# Hoặc set biến môi trường MODEL_PATH khi chạy: export MODEL_PATH="..."

# Option 1: Dùng finetuned_model
# MODEL_PATH = os.path.join(MODEL_BASE_PATH, "finetuned_model")

# Option 2: Dùng checkpoint-3175 (checkpoint cuối cùng)
MODEL_PATH = os.path.join(MODEL_BASE_PATH, "outputs/checkpoint-3175")

# Option 3: Dùng checkpoint khác (uncomment và sửa số checkpoint)
# MODEL_PATH = os.path.join(MODEL_BASE_PATH, "outputs/checkpoint-3000")
# MODEL_PATH = os.path.join(MODEL_BASE_PATH, "outputs/checkpoint-2500")
# MODEL_PATH = os.path.join(MODEL_BASE_PATH, "outputs/checkpoint-2000")
# MODEL_PATH = os.path.join(MODEL_BASE_PATH, "outputs/checkpoint-1500")
# MODEL_PATH = os.path.join(MODEL_BASE_PATH, "outputs/checkpoint-1000")
# MODEL_PATH = os.path.join(MODEL_BASE_PATH, "outputs/checkpoint-500")

# Override bằng biến môi trường nếu có
if os.getenv("MODEL_PATH"):
    env_model_path = os.getenv("MODEL_PATH")
    if os.path.isabs(env_model_path):
        MODEL_PATH = env_model_path
    else:
        MODEL_PATH = os.path.join(MODEL_BASE_PATH, env_model_path)
    logger.info(f"Using MODEL_PATH from environment: {MODEL_PATH}")

# Kiểm tra model path có tồn tại không
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model path không tồn tại: {MODEL_PATH}\n"
        f"Hãy kiểm tra lại đường dẫn hoặc chọn một trong các checkpoint có sẵn."
    )

# ============================================
# CHECK GPU MEMORY AND PROCESSES
# ============================================
def check_gpu_memory_and_processes():
    """Kiểm tra GPU memory và các process đang sử dụng GPU"""
    if not torch.cuda.is_available():
        logger.warning("⚠️ CUDA không khả dụng!")
        return
    
    try:
        import subprocess
        # Lấy thông tin từ nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stdout:
            used_mb, total_mb = map(int, result.stdout.strip().split(", "))
            used_gb = round(used_mb / 1024, 2)
            total_gb = round(total_mb / 1024, 2)
            free_gb = round(total_gb - used_gb, 2)
            usage_percent = round((used_mb / total_mb) * 100, 1)
            
            logger.info(f"📊 GPU Memory Status:")
            logger.info(f"   Total: {total_gb} GB")
            logger.info(f"   Used: {used_gb} GB ({usage_percent}%)")
            logger.info(f"   Free: {free_gb} GB")
            
            # Kiểm tra các process đang sử dụng GPU
            result_processes = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            
            if result_processes.stdout.strip():
                logger.warning("⚠️ Có các process khác đang sử dụng GPU:")
                for line in result_processes.stdout.strip().split("\n"):
                    if line.strip():
                        pid, name, mem = line.split(", ")
                        mem_gb = round(int(mem.replace("MiB", "")) / 1024, 2)
                        logger.warning(f"   PID {pid}: {name} ({mem_gb} GB)")
                        # Kiểm tra xem có phải fastapi_server khác không
                        if "python" in name.lower() or "fastapi" in name.lower():
                            try:
                                # Kiểm tra command line của process
                                proc_info = subprocess.run(
                                    ["ps", "-p", pid, "-o", "cmd="],
                                    capture_output=True,
                                    text=True,
                                    check=True
                                )
                                if "fastapi_server" in proc_info.stdout:
                                    logger.error(f"❌ Phát hiện instance khác của fastapi_server.py (PID {pid}) đang chạy!")
                                    logger.error(f"   Hãy kill process này trước: kill {pid}")
                                    logger.error(f"   Hoặc kill tất cả: pkill -f fastapi_server.py")
                            except:
                                pass
                
                if free_gb < 10:
                    logger.error("❌ GPU memory không đủ để load model!")
                    logger.error("   Cần ít nhất ~10 GB free để load model 20B với 4-bit quantization")
                    logger.error("   Hãy kill các process đang chiếm GPU trước khi chạy lại!")
                    raise RuntimeError(
                        f"GPU memory không đủ! Free: {free_gb} GB, cần ~10 GB. "
                        f"Có process khác đang chiếm GPU. Hãy kill chúng trước."
                    )
            else:
                logger.info("✅ Không có process nào đang sử dụng GPU")
                
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠️ Không thể kiểm tra GPU processes: {e}")
    except Exception as e:
        logger.warning(f"⚠️ Lỗi khi kiểm tra GPU: {e}")

# ============================================
# CLEAR GPU MEMORY
# ============================================
def clear_gpu_memory():
    """Clear GPU cache trước khi load model"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    gc.collect()
    logger.info("GPU memory cleared!")
    
    # Log GPU memory status
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        total_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        reserved_memory = round(torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024, 3)
        allocated_memory = round(torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024, 3)
        free_memory = total_memory - reserved_memory
        logger.info(f"GPU Memory - Total: {total_memory} GB, Reserved: {reserved_memory} GB, Allocated: {allocated_memory} GB, Free: {free_memory} GB")

# Clear compiled cache nếu có (fixes TorchRuntimeError)
cache_dir = os.path.join(MODEL_BASE_PATH, "unsloth_compiled_cache")
if os.path.exists(cache_dir):
    try:
        shutil.rmtree(cache_dir)
        logger.info("✅ Đã xóa compiled cache")
    except Exception as e:
        logger.warning(f"⚠️ Không thể xóa cache: {e}")

# ============================================
# LOAD MODEL
# ============================================
logger.info("=" * 60)
logger.info("Đang load GPT-OSS 20B Finetuned Model...")
logger.info(f"Model path: {MODEL_PATH}")
logger.info("=" * 60)

# Kiểm tra GPU memory và processes trước khi load
check_gpu_memory_and_processes()
clear_gpu_memory()

try:
    # Đảm bảo model được load hoàn toàn trên GPU với 4-bit quantization
    # Tính toán max_memory dựa trên GPU available (để lại ~1GB cho overhead)
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        total_memory_gb = gpu_stats.total_memory / 1024 / 1024 / 1024
        # Để lại 1GB cho overhead và system
        max_memory_gb = int(total_memory_gb - 1)
        max_memory = {0: f"{max_memory_gb}GiB"}
        logger.info(f"Setting max_memory to {max_memory} for GPU 0")
    else:
        max_memory = None
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto detection
        load_in_4bit=True,  # 4-bit quantization để tiết kiệm VRAM cho 16GB
        device_map="cuda:0",  # Đảm bảo load trên GPU
        max_memory=max_memory,  # Giới hạn memory usage
    )
    
    # Set model to inference mode
    FastLanguageModel.for_inference(model)
    
    logger.info("✅ Model đã được load thành công!")
    
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        used_memory = round(torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024, 3)
        logger.info(f"GPU: {gpu_stats.name}")
        logger.info(f"Max memory: {max_memory} GB")
        logger.info(f"Used memory: {used_memory} GB")
    
except Exception as e:
    logger.error(f"❌ Lỗi khi load model: {e}")
    import traceback
    traceback.print_exc()
    raise

# ============================================
# LIFESPAN HANDLER
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler cho startup và shutdown"""
    # Startup
    logger.info("=" * 60)
    logger.info("GPT-OSS 20B Finetuned API Server")
    logger.info(f"Port: {PORT}")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info("=" * 60)
    yield
    # Shutdown (nếu cần cleanup)
    logger.info("Shutting down server...")

# ============================================
# FASTAPI APP
# ============================================
app = FastAPI(title="GPT-OSS 20B Finetuned API", version="1.0.0", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# REQUEST/RESPONSE MODELS
# ============================================
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    reasoning_effort: Optional[str] = "medium"  # "low", "medium", "high"
    messages: Optional[List[Dict[str, str]]] = None  # Optional chat format

class GenerateResponse(BaseModel):
    response: str
    model: str = "gpt-oss-20b-finetuned"

# ============================================
# ENDPOINTS
# ============================================
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "model": "gpt-oss-20b-finetuned",
        "port": PORT,
        "model_path": MODEL_PATH
    }

@app.get("/health")
async def health():
    """Health check với thông tin model"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        gpu_info = {
            "gpu_name": gpu_stats.name,
            "total_memory_gb": round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3),
            "used_memory_gb": round(torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024, 3),
        }
    
    return {
        "status": "healthy",
        "model": "gpt-oss-20b-finetuned",
        "model_path": MODEL_PATH,
        "gpu": gpu_info
    }

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text từ prompt
    
    Args:
        request: GenerateRequest với prompt và các tham số generation
    
    Returns:
        Generated text response
    """
    try:
        # Nếu có messages (chat format), sử dụng messages
        # Nếu không, sử dụng prompt đơn giản
        if request.messages:
            messages = request.messages
        else:
            # Tạo messages từ prompt đơn giản
            messages = [
                {
                    "role": "developer",
                    "content": (
                        "# Instructions\n\nreasoning language: Vietnamese\n\n"
                        "Bạn là một trợ lý AI chuyên về tư vấn pháp luật Việt Nam. "
                        "Hãy phân tích và trả lời câu hỏi dựa trên các văn bản quy phạm pháp luật hiện hành."
                    )
                },
                {"role": "user", "content": request.prompt}
            ]
        
        # Apply chat template với reasoning_effort
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort=request.reasoning_effort,  # "low", "medium", "high"
        ).to("cuda")
        
        # Generate với các tham số
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response (chỉ lấy phần mới generate, không bao gồm prompt)
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        logger.info(f"Generated {len(generated_text)} characters")
        
        return GenerateResponse(response=generated_text)
        
    except Exception as e:
        logger.error(f"Lỗi khi generate: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi khi generate: {str(e)}")

@app.post("/api/chat")
async def chat(request: GenerateRequest):
    """
    Chat endpoint với format messages chuẩn GPT-OSS Harmony
    
    Args:
        request: GenerateRequest với messages (List[Dict[str, str]])
    
    Returns:
        Generated text response
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages là bắt buộc cho endpoint /api/chat")
    
    try:
        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            request.messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort=request.reasoning_effort,
        ).to("cuda")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        return GenerateResponse(response=generated_text)
        
    except Exception as e:
        logger.error(f"Lỗi khi chat: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi khi chat: {str(e)}")

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

