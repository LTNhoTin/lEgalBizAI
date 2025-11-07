import time
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging

from utils import make_async
from generation.gpt_oss_20b import generate_response, load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = ["*"]  # Cho phép tất cả origins để main.py có thể gọi

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_call = {
    "legalbizai-gpt-oss": make_async(generate_response),
}

DEFAULT_MODEL = "legalbizai-gpt-oss"


@app.on_event("startup")
async def startup_event():
    """
    Load GPT-OSS-20B model khi server khởi động
    """
    import threading
    
    def load_model_background():
        """Load model trong background thread"""
        try:
            logger.info("=" * 60)
            logger.info("Đang load GPT-OSS-20B model trong background...")
            logger.info("Server vẫn có thể nhận request trong khi model đang load")
            logger.info("=" * 60)
            load_model()
            logger.info("=" * 60)
            logger.info("✓ GPT-OSS-20B model đã sẵn sàng!")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"⚠ Lỗi khi load model trong startup: {e}")
            logger.error("Model sẽ được load khi có request đầu tiên")
    
    # Load model trong background thread
    thread = threading.Thread(target=load_model_background, daemon=True)
    thread.start()


@app.post("/generate")
async def generate(request: Request):
    """
    Endpoint để generate response từ GPT-OSS-20B model
    Format request: {"prompt": "your prompt here"}
    Format response: {"response": "generated text"}
    """
    logger.info("Received request at /generate")
    body = await request.json()
    model = body.get("model", DEFAULT_MODEL)
    prompt = body.get("prompt", "")
    
    if not prompt:
        return JSONResponse({"error": "Prompt is required"}, status_code=400)
    
    if model not in llm_call:
        return JSONResponse({"error": f"Model {model} not found"}, status_code=400)
    
    start_time = time.time()
    try:
        result = await llm_call[model](prompt)
        execution_time = time.time() - start_time
        logger.info(f"Executed time: {execution_time:.4f} seconds")
        
        return JSONResponse({"response": result})
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({"status": "ok", "model": DEFAULT_MODEL})


if __name__ == "__main__":
    import uvicorn
    
    # Chạy trên port 9000 để không conflict với main.py (port 8000)
    uvicorn.run(app, host="127.0.0.1", port=9000, log_level="info")

