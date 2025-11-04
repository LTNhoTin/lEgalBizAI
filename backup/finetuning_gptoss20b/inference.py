#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, sys, threading, torch
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer

MODEL_PATH = "./models/finetuned_model"
MAX_SEQ_LEN = 4096

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LEN,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
print("Model loaded.")

def chat_once(prompt: str):
    messages = [
        {"role": "user", "content": prompt}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    def worker():
        with torch.no_grad():
            model.generate(**inputs, **kwargs)

    threading.Thread(target=worker, daemon=True).start()

    output = []
    for piece in streamer:
        print(piece, end="", flush=True)   # stream ra ngay
        output.append(piece)
    print()
    return "".join(output)

while True:
    try:
        q = input("\nYou: ").strip()
        if q.lower() in ("/quit", "exit"): break
        print("Assistant: ", end="", flush=True)
        chat_once(q)
    except KeyboardInterrupt:
        break
