#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import uvicorn
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
import threading
import time

# Configuration
MODEL_PATH = "./models/finetuned_model"
MAX_SEQ_LEN = 4096
API_KEY = "gptoss-api-key-2024"

# FastAPI app
app = FastAPI(
    title="Fine-tuned GPT-OSS-20B API",
    description="API server for fine-tuned GPT-OSS-20B model",
    version="1.0.0"
)

# Security
security = HTTPBearer()

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95

class ChatResponse(BaseModel):
    response: str
    model: str = "finetuned-gptoss20b"

# Global variables for model
model = None
tokenizer = None

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key authentication"""
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return credentials.credentials

def load_model():
    """Load the fine-tuned model with 4-bit quantization"""
    global model, tokenizer
    
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LEN,
        dtype=torch.bfloat16,
        load_in_4bit=True,  # Sử dụng 4-bit để tiết kiệm VRAM
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded successfully.")

def generate_response(prompt: str, max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.95) -> str:
    """Generate response from the model"""
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    return response.strip()

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Fine-tuned GPT-OSS-20B API Server",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """Chat endpoint with API key authentication"""
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        response = generate_response(
            prompt=request.message,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return ChatResponse(response=response)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )

@app.get("/info")
async def api_info(api_key: str = Depends(verify_api_key)):
    """Get API information (requires authentication)"""
    return {
        "api_key_valid": True,
        "model_name": "finetuned-gptoss20b",
        "max_sequence_length": MAX_SEQ_LEN,
        "quantization": "4-bit",
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    print(f"\n🔑 API Key: {API_KEY}")
    print("\n📖 API Documentation: http://localhost:7777/docs")
    print("\n🌐 Production URL: https://nhotin.space:7777")
    print("\nStarting FastAPI server...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7777,
        log_level="info"
    )