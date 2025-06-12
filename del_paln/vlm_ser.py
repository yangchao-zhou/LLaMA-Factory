from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
import httpx
import json
import asyncio
from typing import AsyncGenerator

app = FastAPI()

# vLLM 服务地址
VLLM_API_URL = "http://0.0.0.0:3280"

async def process_stream(response: httpx.Response, endpoint=None) -> AsyncGenerator[bytes, None]:
    """处理流式响应，检查 ### Direct Answer 标记"""
    buffer = ""
    complete_content = ""  # 用于累积完整内容
    direct_answer_mode = False
    
    async for chunk in response.aiter_bytes(chunk_size=1024):
        buffer += chunk.decode('utf-8', errors='ignore')
        
        while buffer:
            try:
                obj, idx = json.JSONDecoder().raw_decode(buffer)
                buffer = buffer[idx:].lstrip()
                
                if 'choices' in obj and obj['choices']:
                    if endpoint == "/v1/completions":
                        content = obj.get('choices', [{}])[0].get('text', '')
                    else:
                        content = obj['choices'][0].get('message', {}).get('content', '')
                    
                    # 累积完整内容
                    complete_content += content
                    print(f"complete_content: {complete_content}")
                    # 检查完整内容中是否包含 ### Direct Answer
                    if not direct_answer_mode and '### Direct Answer' in complete_content:
                        direct_answer_mode = True
                        # 从完整内容中提取 Direct Answer 后的部分
                        parts = complete_content.split('### Direct Answer', 1)
                        if len(parts) > 1:
                            if endpoint == "/v1/completions":
                                obj['choices'][0]['text'] = parts[1].lstrip()
                            else:
                                obj['choices'][0]['message']['content'] = parts[1].lstrip()
                    
                    if direct_answer_mode:
                        yield json.dumps(obj, ensure_ascii=False).encode('utf-8') + b'\n'
                    else:
                        yield json.dumps(obj, ensure_ascii=False).encode('utf-8') + b'\n'
                        
            except json.JSONDecodeError:
                break

async def proxy_streaming_request(request: Request, endpoint: str) -> StreamingResponse:
    """代理流式请求，处理 chat/completions 和 completions 接口"""
    async with httpx.AsyncClient(timeout=60.0*10) as client:
        response = await client.request(
            method=request.method,
            url=f"{VLLM_API_URL}{endpoint}",
            headers={k: v for k, v in request.headers.items() if k.lower() != 'host'},
            content=await request.body(),
        )
        
        return StreamingResponse(
            content=process_stream(response, endpoint),
            status_code=response.status_code,
            headers={k: v for k, v in response.headers.items() if k.lower() not in ('transfer-encoding', 'content-length')},
            media_type="application/json"
        )

async def proxy_direct_request(request: Request, endpoint: str) -> Response:
    """直接代理其他接口，保持原样返回"""
    async with httpx.AsyncClient(timeout=60.0*10) as client:
        response = await client.request(
            method=request.method,
            url=f"{VLLM_API_URL}{endpoint}",
            headers={k: v for k, v in request.headers.items() if k.lower() != 'host'},
            content=await request.body(),
        )
        
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers={k: v for k, v in response.headers.items() if k.lower() != 'transfer-encoding'},
            media_type=response.headers.get('content-type')
        )

# 特定接口：处理 ### Direct Answer
@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    return await proxy_streaming_request(request, "/v1/chat/completions")

@app.post("/v1/completions")
async def proxy_completions(request: Request):
    return await proxy_streaming_request(request, "/v1/completions")

# 通用路由：直接透传其他接口
@app.api_route("/{path:path}", methods=["GET", "POST", "HEAD"])
async def catch_all(request: Request, path: str):
    return await proxy_direct_request(request, f"/{path}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)