#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
from typing import Optional

# Configuration
API_BASE_URL = "http://localhost:7777"
API_KEY = "gptoss-api-key-2024"  # Hardcoded API key

class GPTOSSClient:
    """Client để test API server"""
    
    def __init__(self, base_url: str = API_BASE_URL, api_key: str = API_KEY):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
    
    def health_check(self) -> dict:
        """Test health endpoint (no auth required)"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_api_info(self) -> dict:
        """Test API info endpoint"""
        try:
            response = requests.get(
                f"{self.base_url}/info",
                headers=self.headers
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def chat(self, message: str, max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.95) -> dict:
        """Gửi tin nhắn đến model"""
        try:
            payload = {
                "message": message,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            
            response = requests.post(
                f"{self.base_url}/chat",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"HTTP {response.status_code}",
                    "detail": response.text
                }
                
        except Exception as e:
            return {"error": str(e)}

def main():
    """Test client chính"""
    client = GPTOSSClient()
    
    print("🔍 Testing API Server...\n")
    
    # Test health check
    print("1. Health Check:")
    health = client.health_check()
    print(json.dumps(health, indent=2))
    print()
    
    # Test API info
    print("2. API Info:")
    api_info = client.get_api_info()
    print(json.dumps(api_info, indent=2))
    print()
    
    # Test chat
    print("3. Chat Test:")
    test_messages = [
        "Xin chào! Bạn có thể giúp tôi không?",
        "Hãy viết một email marketing cho sản phẩm mới.",
        "Tôi cần lời khuyên về chiến lược bán hàng."
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n3.{i} Testing message: '{message}'")
        result = client.chat(message)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Response: {result['response'][:200]}..." if len(result['response']) > 200 else f"Response: {result['response']}")
    
    print("\n🎯 Interactive Chat Mode (type 'quit' to exit):")
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            if not user_input:
                continue
                
            print("Assistant: ", end="", flush=True)
            result = client.chat(user_input)
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(result['response'])
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()