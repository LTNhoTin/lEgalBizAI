#!/usr/bin/env python3
"""
Script để test inference với model đã train
"""
import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path: str, base_model_name: str = "pansophic/rocket-3B"):
    """
    Load model và tokenizer
    
    Args:
        model_path: Path tới LoRA adapter
        base_model_name: Tên base model
    """
    logger.info(f"Loading base model: {base_model_name}")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    logger.info(f"Loading LoRA adapter from: {model_path}")
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Merge LoRA weights (optional, for faster inference)
    # model = model.merge_and_unload()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    logger.info("Model loaded successfully!")
    
    return model, tokenizer


def generate_answer(
    model,
    tokenizer,
    question: str,
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50
):
    """
    Generate answer cho câu hỏi
    
    Args:
        model: Model
        tokenizer: Tokenizer
        question: Câu hỏi
        max_length: Max length của response
        temperature: Temperature cho sampling
        top_p: Top-p sampling
        top_k: Top-k sampling
    """
    # Format prompt
    prompt = f"""### Câu hỏi:
{question}

### Trả lời:
"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer (remove prompt)
    answer = response.split("### Trả lời:")[-1].strip()
    
    return answer


def interactive_mode(model, tokenizer):
    """Interactive mode để test model"""
    logger.info("\n" + "="*80)
    logger.info("INTERACTIVE MODE")
    logger.info("="*80)
    logger.info("Nhập câu hỏi để test model. Gõ 'quit' hoặc 'exit' để thoát.")
    logger.info("="*80 + "\n")
    
    while True:
        question = input("\n💬 Câu hỏi: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            logger.info("Thoát interactive mode.")
            break
        
        if not question:
            continue
        
        logger.info("\n🤖 Generating answer...")
        
        try:
            answer = generate_answer(model, tokenizer, question)
            
            print("\n" + "="*80)
            print("📝 ANSWER:")
            print("="*80)
            print(answer)
            print("="*80)
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")


def main():
    parser = argparse.ArgumentParser(description='Inference với model đã train')
    parser.add_argument(
        '--model-path',
        type=str,
        default='outputs/final_model',
        help='Path tới LoRA adapter (default: outputs/final_model)'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default='pansophic/rocket-3B',
        help='Tên base model (default: pansophic/rocket-3B)'
    )
    parser.add_argument(
        '--question',
        type=str,
        default=None,
        help='Câu hỏi để test (nếu không có sẽ vào interactive mode)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Max length cho response (default: 512)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Temperature (default: 0.7)'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found: {args.model_path}")
        logger.error("Train model trước bằng: python main.py")
        sys.exit(1)
    
    # Load model
    logger.info("Loading model...")
    model, tokenizer = load_model(args.model_path, args.base_model)
    
    # Inference
    if args.question:
        # Single question mode
        logger.info(f"\nCâu hỏi: {args.question}")
        logger.info("\nGenerating answer...")
        
        answer = generate_answer(
            model, 
            tokenizer, 
            args.question,
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(answer)
        print("="*80 + "\n")
    else:
        # Interactive mode
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()


