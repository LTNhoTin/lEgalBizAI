#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module để load và sử dụng GPT-OSS-20B finetuned model từ checkpoint
"""

import os
import torch
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
import threading

# Đường dẫn tới checkpoint
CHECKPOINT_PATH = "/home/nhotin/work/LegalBizAI_project/finetune/gpt_oss_20b_finetune/outputs/checkpoint-3175"
MAX_SEQ_LEN = 4096

# Global variables để cache model và tokenizer
_model = None
_tokenizer = None
_model_loaded = False

def load_model():
    """Load GPT-OSS-20B finetuned model từ checkpoint"""
    global _model, _tokenizer, _model_loaded
    
    if _model_loaded:
        print("Model đã được load trước đó, sử dụng model đã cache.")
        return _model, _tokenizer
    
    print(f"Đang load GPT-OSS-20B finetuned model từ {CHECKPOINT_PATH}...")
    
    try:
        # Load model từ checkpoint (unsloth tự động load base model và adapter)
        _model, _tokenizer = FastLanguageModel.from_pretrained(
            model_name=CHECKPOINT_PATH,
            max_seq_length=MAX_SEQ_LEN,
            dtype=torch.bfloat16,
            load_in_4bit=True,  # Sử dụng 4-bit để tiết kiệm VRAM
        )
        
        # Chuyển sang inference mode
        FastLanguageModel.for_inference(_model)
        
        _model_loaded = True
        print("Model đã được load thành công!")
        
    except Exception as e:
        print(f"Lỗi khi load model: {e}")
        raise
    
    return _model, _tokenizer

def generate_response(prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.95) -> str:
    """
    Generate response từ prompt sử dụng GPT-OSS-20B finetuned model
    
    Args:
        prompt: Câu hỏi hoặc prompt đầu vào
        max_new_tokens: Số token tối đa để generate
        temperature: Temperature cho sampling
        top_p: Top-p sampling parameter
    
    Returns:
        Generated text response
    """
    global _model, _tokenizer
    
    # Đảm bảo model đã được load
    if not _model_loaded:
        _model, _tokenizer = load_model()
    
    # Format prompt thành messages
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template
    inputs = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(_model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )
    
    # Decode response (chỉ lấy phần generated, không bao gồm prompt)
    response = _tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )[0].strip()
    
    return response

def generate_response_streaming(prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.95):
    """
    Generate response với streaming (yield từng chunk)
    
    Args:
        prompt: Câu hỏi hoặc prompt đầu vào
        max_new_tokens: Số token tối đa để generate
        temperature: Temperature cho sampling
        top_p: Top-p sampling parameter
    
    Yields:
        Generated text chunks
    """
    global _model, _tokenizer
    
    # Đảm bảo model đã được load
    if not _model_loaded:
        _model, _tokenizer = load_model()
    
    # Format prompt thành messages
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template
    inputs = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(_model.device)
    
    # Setup streamer
    streamer = TextIteratorStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=_tokenizer.eos_token_id,
        streamer=streamer,
    )
    
    # Generate trong thread riêng
    def worker():
        with torch.no_grad():
            _model.generate(**inputs, **kwargs)
    
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    
    # Yield từng chunk
    for piece in streamer:
        yield piece

# Load model khi import module (lazy loading có thể được thêm sau nếu cần)
# load_model()

