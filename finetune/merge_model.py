#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để merge LoRA adapter với base model và lưu thành model đầy đủ
"""

import os
import torch
from unsloth import FastLanguageModel

# Set HuggingFace token từ environment variable
# Sử dụng: export HF_TOKEN=your_token_here trước khi chạy script
if "HF_TOKEN" not in os.environ:
    print("Warning: HF_TOKEN not set. Please set it using: export HF_TOKEN=your_token_here")

# Đường dẫn
CHECKPOINT_PATH = "/home/nhotin/work/LegalBizAI_project/finetune/gpt_oss_20b_finetune/outputs/checkpoint-3175"
OUTPUT_PATH = "/home/nhotin/work/LegalBizAI_project/finetune/gpt_oss_20b_finetune/merged_model"

MAX_SEQ_LEN = 4096

def merge_and_save_model():
    """
    Merge LoRA adapter với base model và lưu thành model đầy đủ
    """
    print("=" * 80)
    print("MERGE LORA ADAPTER VỚI BASE MODEL")
    print("=" * 80)
    
    print(f"\n📂 Checkpoint path: {CHECKPOINT_PATH}")
    print(f"💾 Output path: {OUTPUT_PATH}")
    
    # Bước 1: Load model và adapter với Unsloth
    print("\n⏳ Bước 1: Load base model và LoRA adapter...")
    print("   (Load trên GPU với 4bit quantization)")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=CHECKPOINT_PATH,
            max_seq_length=MAX_SEQ_LEN,
            dtype=None,  # Auto detect
            load_in_4bit=True,  # Dùng 4bit để tiết kiệm memory
            device_map="auto",  # Load trên GPU
        )
        print("✅ Load thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi load model: {e}")
        raise
    
    # Bước 2: Merge LoRA adapter vào base model
    print("\n⏳ Bước 2: Merge LoRA adapter vào base model...")
    try:
        # Merge adapter weights vào base model
        model = model.merge_and_unload()
        print("✅ Merge thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi merge: {e}")
        raise
    
    # Bước 3: Lưu merged model
    print(f"\n⏳ Bước 3: Lưu merged model vào {OUTPUT_PATH}...")
    try:
        # Tạo thư mục output nếu chưa có
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        
        # Lưu model và tokenizer
        model.save_pretrained(OUTPUT_PATH)
        tokenizer.save_pretrained(OUTPUT_PATH)
        
        print("✅ Lưu thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi lưu model: {e}")
        raise
    
    print("\n" + "=" * 80)
    print("🎉 HOÀN THÀNH!")
    print("=" * 80)
    print(f"\nModel đã được lưu tại: {OUTPUT_PATH}")
    print("\nBạn có thể sử dụng model này bằng cách:")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{OUTPUT_PATH}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{OUTPUT_PATH}')")
    
    # Hiển thị kích thước model
    try:
        import subprocess
        result = subprocess.run(['du', '-sh', OUTPUT_PATH], capture_output=True, text=True)
        if result.returncode == 0:
            size = result.stdout.split()[0]
            print(f"\n📦 Kích thước model: {size}")
    except:
        pass

if __name__ == "__main__":
    merge_and_save_model()

