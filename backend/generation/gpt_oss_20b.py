#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module để load và sử dụng GPT-OSS-20B finetuned model từ checkpoint
"""

import os
import re
import torch
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
import threading

# Set HuggingFace token từ environment variable
# Sử dụng: export HF_TOKEN=your_token_here trước khi chạy server
if "HF_TOKEN" not in os.environ:
    print("Warning: HF_TOKEN not set. Please set it using: export HF_TOKEN=your_token_here")

# Đường dẫn tới checkpoint
CHECKPOINT_PATH = "/home/nhotin/work/LegalBizAI_project/finetune/gpt_oss_20b_finetune/outputs/checkpoint-3175"
MAX_SEQ_LEN = 4096

# Global variables để cache model và tokenizer
_model = None
_tokenizer = None
_model_loaded = False

def _extract_final_answer(response: str) -> str:
    """
    Extract phần câu trả lời cuối cùng từ response.
    Nếu response có format "analysis...final...", chỉ lấy phần sau "final"
    
    GPT-OSS Harmony format có thể generate:
    - "analysisCâu hỏi...finalCâu trả lời..." (không có space)
    - "analysis Câu hỏi...final Câu trả lời..." (có space)
    
    Args:
        response: Raw response từ model
    
    Returns:
        Cleaned response (chỉ phần final answer, bỏ phần analysis)
    """
    # Tìm vị trí của "final" trong response (case-insensitive)
    # GPT-OSS Harmony format: "analysisCâu hỏi...finalCâu trả lời..."
    
    # Tìm "final" - có thể là "final", "Final", "FINAL"
    final_idx = -1
    for pattern in ['final', 'Final', 'FINAL']:
        idx = response.find(pattern)
        if idx != -1:
            final_idx = idx
            break
    
    if final_idx != -1:
        # Lấy phần sau "final" (bỏ qua "final" + 5 ký tự)
        answer = response[final_idx + 5:].strip()
        if answer:  # Nếu có nội dung sau "final"
            return answer
    
    # Nếu không tìm thấy "final", kiểm tra xem có "analysis" ở đầu không
    # Nếu có "analysis" mà không có "final", có thể model đang thinking
    if response.lower().startswith('analysis'):
        # Model đang trong quá trình reasoning, chưa có final answer
        # Trong trường hợp này, trả về message hoặc empty
        # hoặc có thể return response gốc tùy use case
        pass
    
    # Nếu không match pattern nào, return nguyên response
    return response

def load_model():
    """Load GPT-OSS-20B finetuned model từ checkpoint sử dụng Unsloth"""
    global _model, _tokenizer, _model_loaded
    
    if _model_loaded:
        print("Model đã được load trước đó, sử dụng model đã cache.")
        return _model, _tokenizer
    
    print(f"Đang load GPT-OSS-20B finetuned model từ {CHECKPOINT_PATH} với Unsloth...")
    
    try:
        # Load model và tokenizer với FastLanguageModel từ Unsloth
        # Use device_map="auto" để tự động phân bổ model giữa GPU/CPU
        _model, _tokenizer = FastLanguageModel.from_pretrained(
            model_name=CHECKPOINT_PATH,
            max_seq_length=MAX_SEQ_LEN,
            dtype=None,  # Auto detect
            load_in_4bit=True,  # 4-bit quantization
            device_map="auto",  # Auto distribute model across devices
        )
        
        # Set model to inference mode
        FastLanguageModel.for_inference(_model)
        
        _model_loaded = True
        print("✓ Model đã được load thành công với Unsloth!")
        
    except Exception as e:
        print(f"⚠ Lỗi khi load model: {e}")
        raise
    
    return _model, _tokenizer

def generate_response(prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.95, reasoning_effort: str = "low") -> str:
    """
    Generate response từ prompt sử dụng GPT-OSS-20B finetuned model
    
    Args:
        prompt: Câu hỏi hoặc prompt đầu vào
        max_new_tokens: Số token tối đa để generate
        temperature: Temperature cho sampling
        top_p: Top-p sampling parameter
        reasoning_effort: "low", "medium", hoặc "high" - mức độ reasoning
    
    Returns:
        Generated text response
    """
    global _model, _tokenizer
    
    # Đảm bảo model đã được load
    if not _model_loaded:
        _model, _tokenizer = load_model()
    
    # Format prompt thành messages theo format GPT-OSS Harmony
    # Giống như trong notebook finetune
    messages = [
        {
            "role": "developer",
            "content": (
                "# Instructions\n\nreasoning language: Vietnamese\n\n"
                "Bạn là một trợ lý AI chuyên về tư vấn pháp luật Việt Nam. "
                "Hãy phân tích và trả lời câu hỏi dựa trên các văn bản quy phạm pháp luật hiện hành. "
                "Câu trả lời phải chính xác, có căn cứ pháp lý rõ ràng và dễ hiểu."
            )
        },
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template với reasoning_effort
    inputs = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort=reasoning_effort,  # Thêm reasoning effort
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
    
    # Post-process: Nếu response có format "analysis...final...", chỉ lấy phần sau "final"
    response = _extract_final_answer(response)
    
    return response

def generate_response_streaming(prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.95, reasoning_effort: str = "medium"):
    """
    Generate response với streaming (yield từng chunk)
    
    Args:
        prompt: Câu hỏi hoặc prompt đầu vào
        max_new_tokens: Số token tối đa để generate
        temperature: Temperature cho sampling
        top_p: Top-p sampling parameter
        reasoning_effort: "low", "medium", hoặc "high" - mức độ reasoning
    
    Yields:
        Generated text chunks
    """
    global _model, _tokenizer
    
    # Đảm bảo model đã được load
    if not _model_loaded:
        _model, _tokenizer = load_model()
    
    # Format prompt thành messages theo format GPT-OSS Harmony
    messages = [
        {
            "role": "developer",
            "content": (
                "# Instructions\n\nreasoning language: Vietnamese\n\n"
                "Bạn là một trợ lý AI chuyên về tư vấn pháp luật Việt Nam. "
                "Hãy phân tích và trả lời câu hỏi dựa trên các văn bản quy phạm pháp luật hiện hành. "
                "Câu trả lời phải chính xác, có căn cứ pháp lý rõ ràng và dễ hiểu."
            )
        },
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template với reasoning_effort
    inputs = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort=reasoning_effort,
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

