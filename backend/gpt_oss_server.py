#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI server để host GPT-OSS-20B model
"""

import os
import time
import logging
import threading
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from generation.gpt_oss_20b import load_model, generate_response, generate_response_streaming, CHECKPOINT_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GPT-OSS-20B Server", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = "gpt-oss-20b"
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False
    reasoning_effort: Optional[str] = "medium"  # "low", "medium", or "high"


class GenerateResponse(BaseModel):
    model: str
    response: str
    created_at: str
    done: bool = True


@app.on_event("startup")
async def startup_event():
    """
    Load GPT-OSS-20B finetuned model khi server khởi động
    """
    def load_model_background():
        """Load model trong background thread"""
        try:
            logger.info("=" * 60)
            logger.info("Đang load GPT-OSS-20B finetuned model trong background...")
            logger.info(f"Checkpoint path: {CHECKPOINT_PATH}")
            
            # Kiểm tra checkpoint có tồn tại không
            if not os.path.exists(CHECKPOINT_PATH):
                logger.warning(f"⚠ Checkpoint không tồn tại tại: {CHECKPOINT_PATH}")
                logger.warning("Server vẫn sẽ cố gắng load model...")
            else:
                logger.info("✓ Checkpoint path hợp lệ")
            
            logger.info("Server vẫn có thể nhận request trong khi model đang load")
            logger.info("=" * 60)
            load_model()
            logger.info("=" * 60)
            logger.info("✓ GPT-OSS-20B finetuned model đã sẵn sàng!")
            logger.info(f"✓ Đang sử dụng checkpoint: {CHECKPOINT_PATH}")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"⚠ Lỗi khi load model trong startup: {e}")
            logger.error(f"Checkpoint path: {CHECKPOINT_PATH}")
            logger.error("Model sẽ được load khi có request đầu tiên")
    
    # Load model trong background thread
    thread = threading.Thread(target=load_model_background, daemon=True)
    thread.start()


@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """
    Endpoint để generate response từ GPT-OSS-20B model
    Tương tự Ollama API
    """
    logger.info(f"Received generate request: prompt length={len(request.prompt)}")
    
    if not request.prompt:
        return JSONResponse({"error": "Prompt is required"}, status_code=400)
    
    start_time = time.time()
    
    try:
        if request.stream:
            # Streaming response
            def stream_generator():
                try:
                    for chunk in generate_response_streaming(
                        prompt=request.prompt,
                        max_new_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        reasoning_effort=request.reasoning_effort
                    ):
                        yield chunk
                except Exception as e:
                    logger.error(f"Error in streaming: {e}")
                    yield f"\n[Error: {str(e)}]"
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/plain"
            )
        else:
            # Non-streaming response
            result = generate_response(
                prompt=request.prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                reasoning_effort=request.reasoning_effort
            )
            
            execution_time = time.time() - start_time
            logger.info(f"Generated response in {execution_time:.4f} seconds")
            
            return JSONResponse({
                "model": request.model,
                "response": result,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "done": True
            })
    
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/chat")
async def chat(request: Request):
    """
    Endpoint chat tương tự Ollama
    Format: {"model": "gpt-oss-20b", "messages": [{"role": "user", "content": "..."}]}
    """
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "gpt-oss-20b")
    stream = body.get("stream", False)
    
    if not messages:
        return JSONResponse({"error": "Messages are required"}, status_code=400)
    
    # Lấy prompt từ message cuối cùng
    prompt = messages[-1].get("content", "")
    if not prompt:
        return JSONResponse({"error": "Message content is required"}, status_code=400)
    
    # Forward to generate endpoint
    generate_request = GenerateRequest(
        prompt=prompt,
        model=model,
        stream=stream,
        max_tokens=body.get("max_tokens", 512),
        temperature=body.get("temperature", 0.7),
        top_p=body.get("top_p", 0.95),
        reasoning_effort=body.get("reasoning_effort", "medium")
    )
    
    return await generate(generate_request)


@app.get("/api/tags")
async def list_models():
    """
    List available models (tương tự Ollama)
    """
    return JSONResponse({
        "models": [
            {
                "name": "gpt-oss-20b",
                "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "size": 0,  # Có thể thêm size thực tế nếu cần
                "digest": "",
                "details": {
                    "parent_model": "",
                    "format": "pytorch",
                    "family": "gpt-oss",
                    "parameter_size": "20B"
                }
            }
        ]
    })


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "ok",
        "model": "gpt-oss-20b"
    })


@app.get("/")
async def root():
    """Root endpoint"""
    return JSONResponse({
        "message": "GPT-OSS-20B Finetuned Server",
        "version": "1.0.0",
        "checkpoint": CHECKPOINT_PATH,
        "endpoints": {
            "generate": "/api/generate",
            "chat": "/api/chat",
            "models": "/api/tags",
            "health": "/health"
        }
    })


if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 60)
    logger.info("Starting GPT-OSS-20B Finetuned Server")
    logger.info(f"Checkpoint: {CHECKPOINT_PATH}")
    logger.info("Server URL: http://127.0.0.1:1143")
    logger.info("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=1143, log_level="info")
