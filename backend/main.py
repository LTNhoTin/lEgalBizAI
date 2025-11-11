import json
import httpx
import os
import logging
from fastapi.middleware.cors import CORSMiddleware
from utils import RetrieveContent
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from retrieval.retrieve import retrieve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gemini API configuration cho LegalBizAI Pro
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"

# Ollama GPT-OSS-20B configuration cho LegalBizAI
OLLAMA_SERVER_URL = os.getenv("OLLAMA_SERVER_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("Starting LegalBizAI Chatbot Server")
    logger.info("=" * 60)
    
    # Kiểm tra retriever đã được load
    try:
        from retrieval.retrieve import faiss_index, model
        if faiss_index is None:
            logger.warning("⚠ FAISS index chưa được load")
        else:
            logger.info(f"  - Embedding model: {type(model).__name__}")
            logger.info(f"  - FAISS index shape: {faiss_index.ntotal} vectors")
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra retriever: {e}")
    
    # Log cấu hình
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info(f"  - LegalBizAI (Ollama GPT-OSS-20B): {OLLAMA_SERVER_URL}")
    logger.info(f"  - Ollama Model: {OLLAMA_MODEL}")
    logger.info(f"  - LegalBizAI Pro (Gemini): gemini-2.0-flash-exp")
    if GEMINI_API_KEY:
        logger.info(f"  - Gemini API Key: {'*' * 20}...{GEMINI_API_KEY[-4:]}")
    else:
        logger.warning("  - Gemini API Key: CHƯA ĐƯỢC CẤU HÌNH")
    logger.info("=" * 60)

async def stream_get_answer(model, prompt):
    """
    Generate answer từ prompt dựa trên model được chọn
    
    Args:
        model: "LegalBizAI" hoặc "LegalBizAI_pro"
        prompt: Prompt đầu vào
    
    Returns:
        Generated text response
    """
    if model == "LegalBizAI_pro":
        # Sử dụng Gemini API
        if not GEMINI_API_KEY:
            return "Lỗi: GEMINI_API_KEY chưa được cấu hình. Vui lòng set environment variable GEMINI_API_KEY."
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                logger.info("Đang gửi request đến Gemini API...")
                logger.info(f"URL: {GEMINI_API_URL}")
                logger.info(f"Prompt length: {len(prompt)} characters")
                
                res = await client.post(GEMINI_API_URL, headers=headers, json=payload)
                
                logger.info(f"Gemini API Status Code: {res.status_code}")
                
                res.raise_for_status()
                res_data = res.json()
                
                # Extract response từ Gemini API format
                response_text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                
                if not response_text:
                    logger.warning("Gemini API trả về response rỗng")
                    return "Không có phản hồi từ Gemini API"
                
                logger.info(f"Nhận được response từ Gemini API (length: {len(response_text)} characters)")
                return response_text
                
        except httpx.HTTPStatusError as e:
            error_msg = f"Lỗi HTTP {e.response.status_code} khi gọi Gemini API"
            logger.error(f"{error_msg}")
            logger.error(f"Response Body: {e.response.text}")
            return f"{error_msg}: {e.response.text}"
        except httpx.HTTPError as e:
            error_msg = f"Lỗi khi gọi Gemini API: {str(e)}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Lỗi không xác định khi gọi Gemini API: {str(e)}"
            logger.error(error_msg)
            return error_msg

    elif model == "LegalBizAI":
        # Sử dụng Ollama GPT-OSS-20B model
        try:
            logger.info(f"Gọi Ollama GPT-OSS-20B tại: {OLLAMA_SERVER_URL}")
            logger.info(f"Model: {OLLAMA_MODEL}")
            logger.info(f"Prompt length: {len(prompt)} characters")
            
            headers = {
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "temperature": 0.1,
                "top_p": 0.75,
                "top_k": 10,
                "stream": False  # Không stream để đơn giản
            }
            
            async with httpx.AsyncClient(timeout=180.0) as client:
                logger.info("Đang gửi request đến Ollama server...")
                ollama_api_url = f"{OLLAMA_SERVER_URL}/api/generate"
                res = await client.post(ollama_api_url, headers=headers, json=payload)
                res.raise_for_status()
                res_json = res.json()
                
                # Extract response từ Ollama API format
                response_text = res_json.get("response", "")
                if not response_text:
                    logger.warning("Ollama server trả về response rỗng")
                    return "Không có phản hồi từ Ollama server"
                
                logger.info(f"Nhận được response từ Ollama (length: {len(response_text)} characters)")
                return response_text
                
        except httpx.TimeoutException:
            error_msg = f"Timeout khi gọi Ollama server tại {OLLAMA_SERVER_URL}. Model có thể đang load hoặc prompt quá dài."
            logger.error(error_msg)
            return error_msg
        except httpx.ConnectError:
            error_msg = f"Không thể kết nối đến Ollama server tại {OLLAMA_SERVER_URL}. Vui lòng đảm bảo Ollama đang chạy."
            logger.error(error_msg)
            return error_msg
        except httpx.HTTPStatusError as e:
            error_msg = f"Lỗi HTTP {e.response.status_code} khi gọi Ollama: {e.response.text}"
            logger.error(error_msg)
            return error_msg
        except httpx.HTTPError as e:
            error_msg = f"Lỗi HTTP khi gọi Ollama: {str(e)}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Lỗi không xác định khi gọi Ollama: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg
    
    else:
        return f"Model không hợp lệ: {model}"

# async def stream_get_answer(model, prompt):
#     payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]})
#     headers = {"Content-Type": "application/json"}
    
#     if model == "LegalBizAI":
#         API_URL = 'xxx'

#     elif model == "LegalBizAI_pro":
#         API_URL = 'http://127.0.0.1:9000/chat/'

#     async with httpx.AsyncClient(timeout=None) as client:
#         res = await client.post(API_URL, headers=headers, data=payload)
#         res = res.json()
#         text = res["candidates"][0]["content"]["parts"][0]["text"]
#         return text

#             # async for chunk in response.aiter_bytes():
#             #     yield chunk

#     # return = {
#     #     "candidates":[{"content":{"parts":[{"text": Vistral7b_return}]}}]
#     # }

@app.post("/stream")
async def stream_response(request: Request):
    body = await request.json()
    message = body.get("message", "")
    model = body.get("model", "")
    print(message)
    print(model)
    retrieve_content =  RetrieveContent(message)
    prompt = retrieve_content.get_prompt()
    print(prompt)
    if not prompt:
        return {"error": "Prompt is required"}
    # return StreamingResponse(stream_gemini_api(prompt), media_type="application/json")
    answer = await stream_get_answer(model, prompt)

    referenced_response =  retrieve_content.get_true_references(answer)
    print(referenced_response)
    return referenced_response

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
