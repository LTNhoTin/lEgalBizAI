#!/usr/bin/env python3
"""
Script test LLM API server
"""
import requests
import json
import sys

API_URL = "http://localhost:1143"

def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("🔍 Testing Health Endpoint")
    print("=" * 60)
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        print(json.dumps(data, indent=2, ensure_ascii=False))
        print("✅ Health check passed!\n")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}\n")
        return False

def test_generate(prompt, max_new_tokens=300, temperature=0.7, reasoning_effort="medium"):
    """Test generate endpoint"""
    print("=" * 60)
    print(f"💬 Testing Generate Endpoint")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print(f"Max tokens: {max_new_tokens}, Temperature: {temperature}, Reasoning: {reasoning_effort}")
    print("-" * 60)
    
    try:
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort
        }
        
        response = requests.post(
            f"{API_URL}/api/generate",
            json=payload,
            timeout=120  # Model có thể mất thời gian để generate
        )
        response.raise_for_status()
        data = response.json()
        
        print("\n📝 Response:")
        print("-" * 60)
        print(data["response"])
        print("-" * 60)
        print(f"✅ Generate successful! Model: {data['model']}\n")
        return True
    except requests.exceptions.Timeout:
        print("❌ Request timeout (model đang xử lý quá lâu)\n")
        return False
    except Exception as e:
        print(f"❌ Generate failed: {e}\n")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"Error details: {e.response.json()}")
            except:
                print(f"Error text: {e.response.text}")
        print()
        return False

def main():
    """Main test function"""
    print("\n🚀 Starting LLM API Tests\n")
    
    # Test health
    if not test_health():
        print("❌ Server không hoạt động. Hãy kiểm tra lại!")
        sys.exit(1)
    
    # Test cases
    test_cases = [
        {
            "prompt": "Xin chào, bạn có thể giới thiệu về mình không?",
            "max_new_tokens": 200,
            "temperature": 0.7,
            "reasoning_effort": "low"
        },
        {
            "prompt": "Thời hiệu khởi kiện tranh chấp hợp đồng mua bán nhà ở là bao lâu?",
            "max_new_tokens": 400,
            "temperature": 0.7,
            "reasoning_effort": "medium"
        },
        {
            "prompt": "Điều kiện để được cấp giấy phép kinh doanh là gì?",
            "max_new_tokens": 350,
            "temperature": 0.7,
            "reasoning_effort": "medium"
        },
        {
            "prompt": "Quy định về thời gian làm việc của người lao động theo Bộ luật Lao động 2019?",
            "max_new_tokens": 300,
            "temperature": 0.7,
            "reasoning_effort": "high"
        }
    ]
    
    # Run tests
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}/{len(test_cases)}")
        if test_generate(**test_case):
            passed += 1
        else:
            failed += 1
    
    # Summary
    print("=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    print(f"✅ Passed: {passed}/{len(test_cases)}")
    print(f"❌ Failed: {failed}/{len(test_cases)}")
    print("=" * 60)

if __name__ == "__main__":
    main()

