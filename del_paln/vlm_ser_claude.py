from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
import asyncio
import aiofiles
import logging
from datetime import datetime
from typing import AsyncGenerator
from asyncio import Queue
import uuid
from contextlib import asynccontextmanager

# vLLM 服务地址
VLLM_API_URL = "https://sd0kkieqcirbt02vttd60.apigateway-cn-beijing.volceapi.com"
# VLLM_API_URL = "http://localhost:3280"

# 日志配置
def get_log_file_path():
    """生成带时间戳的日志文件路径"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"chat_logs_{timestamp}.jsonl"

LOG_FILE_PATH = get_log_file_path()
log_queue = Queue()

class AsyncLogger:
    """异步日志记录器，使用队列避免IO阻塞"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.running = True
        
    async def start_logger(self):
        """启动日志记录任务"""
        asyncio.create_task(self._log_worker())
    
    async def _log_worker(self):
        """异步日志工作线程"""
        while self.running:
            try:
                # 等待日志条目，超时1秒
                log_entry = await asyncio.wait_for(log_queue.get(), timeout=1.0)
                await self._write_log(log_entry)
                log_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"日志记录错误: {e}")
    
    async def _write_log(self, log_entry: dict):
        """写入日志到文件"""
        try:
            # 使用标准的异步操作避免类型错误
            import aiofiles
            async with aiofiles.open(self.log_file, 'a', encoding='utf-8') as f:  # type: ignore
                await f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"写入日志文件错误: {e}")
    
    async def log_interaction(self, request_id: str, user_input: str, ai_response: str, endpoint: str, messages=None, prompt=None, is_streaming=None):
        """记录用户交互"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "endpoint": endpoint,
            "is_streaming": is_streaming,
            "user_input": user_input,
            "ai_response": ai_response
           
        }
        
        # 如果是 chat/completions 接口，保存完整的 messages
        if endpoint == "/v1/chat/completions" and messages:
            log_entry["messages"] = messages
        
        # 如果是 completions 接口，保存 prompt
        if endpoint == "/v1/completions" and prompt:
            log_entry["prompt"] = prompt
            
        await log_queue.put(log_entry)
    
    def stop(self):
        """停止日志记录器"""
        self.running = False

# 创建全局日志记录器
logger = AsyncLogger(LOG_FILE_PATH)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理器"""
    # 启动事件
    await logger.start_logger()
    print(f"日志记录器已启动，日志文件: {LOG_FILE_PATH}")
    
    yield
    
    # 关闭事件
    logger.stop()
    # 等待队列中的日志处理完成
    await log_queue.join()
    print("日志记录器已停止")

# 创建 FastAPI 应用，使用 lifespan 管理器
app = FastAPI(lifespan=lifespan)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源，生产环境建议指定具体域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)

async def process_stream(response: httpx.Response, endpoint=None, request_id=None) -> AsyncGenerator[bytes, None]:
    """处理流式响应，收集完整内容后检查 ### Direct Answer:\n 标记"""
    buffer = ""
    complete_content = ""  # 用于累积完整内容
    all_chunks = []  # 存储所有的响应块
    final_response = ""  # 用于记录最终AI回复
    stream_finished = False
    
    print(f"Starting stream processing for endpoint: {endpoint}")
    
    # 第一阶段：收集所有响应内容
    try:
        async for chunk in response.aiter_bytes(chunk_size=1024):
            chunk_text = chunk.decode('utf-8', errors='ignore')
            buffer += chunk_text
            
            # 按行处理流式响应
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()
                
                if not line:
                    continue
                    
                # 处理 Server-Sent Events 格式
                if line.startswith('data: '):
                    json_str = line[6:]  # 移除 'data: ' 前缀
                    
                    if json_str == '[DONE]':
                        stream_finished = True
                        break
                    
                    try:
                        obj = json.loads(json_str)
                        all_chunks.append((obj, True))  # True 表示是 SSE 格式
                        
                        # 调试：打印接收到的数据结构
                        print(f"[DEBUG] Received SSE object for {endpoint}: {obj}")
                        
                        # 提取内容
                        if 'choices' in obj and obj['choices'] and len(obj['choices']) > 0:
                            choice = obj['choices'][0]
                            print(f"[DEBUG] Choice data: {choice}")
                            
                            if endpoint == "/v1/completions":
                                content = choice.get('text', '')
                                print(f"[DEBUG] Completions text content: {repr(content)}")
                            else:
                                delta = choice.get('delta', {})
                                content = delta.get('content', '')
                                print(f"[DEBUG] Chat delta content: {repr(content)}")
                            
                            if content:
                                complete_content += content
                                print(f"[DEBUG] Total content length now: {len(complete_content)}")
                                
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error for SSE: {json_str}, error: {e}")
                        # 保存无法解析的行
                        all_chunks.append((line, True))
                else:
                    # 尝试直接解析为 JSON
                    try:
                        obj = json.loads(line)
                        all_chunks.append((obj, False))  # False 表示不是 SSE 格式
                        
                        # 提取内容
                        if 'choices' in obj and obj['choices'] and len(obj['choices']) > 0:
                            choice = obj['choices'][0]
                            if endpoint == "/v1/completions":
                                content = choice.get('text', '')
                            else:
                                delta = choice.get('delta', {})
                                content = delta.get('content', '')
                            
                            if content:
                                complete_content += content
                                
                    except json.JSONDecodeError:
                        # 保存无法解析的行
                        all_chunks.append((line, False))
            
            if stream_finished:
                break
                
    except Exception as e:
        print(f"Error processing stream: {e}")
    
    print(f"Complete collected content: {repr(complete_content)}")
    print(f"Total chunks collected: {len(all_chunks)}")
    
    # 第二阶段：检查是否包含 ### Direct Answer:\n 并决定如何输出
    has_direct_answer = '### Direct Answer:\n' in complete_content
    if has_direct_answer:
        print("Detected Direct Answer in complete content")
        # 提取 Direct Answer 后的内容
        parts = complete_content.split('### Direct Answer:\n', 1)
        if len(parts) > 1:
            direct_answer_content = parts[1].strip()
            print(f"Direct answer content: {repr(direct_answer_content)}")
            final_response = direct_answer_content
        else:
            direct_answer_content = ""
            final_response = ""
        
        # 重新生成流式响应，只包含 Direct Answer 后的内容
        if direct_answer_content:
            # 将 Direct Answer 内容分解成多个小块来模拟流式输出
            chunk_size = 1  # 每次输出1个字符，保持更自然的流式效果
            for i in range(0, len(direct_answer_content), chunk_size):
                chunk_text = direct_answer_content[i:i+chunk_size]
                
                if endpoint == "/v1/completions":
                    obj = {
                        "choices": [{"text": chunk_text, "index": 0}],
                        "object": "text_completion"
                    }
                else:
                    obj = {
                        "choices": [{"delta": {"content": chunk_text}, "index": 0}],
                        "object": "chat.completion.chunk"
                    }
                
                yield f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode('utf-8')
        
        # 发送结束标记
        if endpoint == "/v1/completions":
            finish_obj = {
                "choices": [{"text": "", "finish_reason": "stop", "index": 0}],
                "object": "text_completion"
            }
        else:
            finish_obj = {
                "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
                "object": "chat.completion.chunk"
            }
        
        # 添加日志记录标记
        if request_id:
            finish_obj['_request_id'] = request_id
            finish_obj['_final_response'] = final_response
            finish_obj['_complete_content'] = complete_content
        
        yield f"data: {json.dumps(finish_obj, ensure_ascii=False)}\n\n".encode('utf-8')
        yield b"data: [DONE]\n\n"
        
    else:
        # 没有 Direct Answer，正常输出所有内容
        print("No Direct Answer detected, outputting original content")
        final_response = complete_content
        
        for chunk_data, is_sse in all_chunks:
            if isinstance(chunk_data, dict):
                # 在最后一个块添加日志记录标记
                if ('choices' in chunk_data and chunk_data['choices'] and 
                    len(chunk_data['choices']) > 0 and 
                    chunk_data['choices'][0].get('finish_reason')):
                    if request_id:
                        chunk_data['_request_id'] = request_id
                        chunk_data['_final_response'] = final_response
                        chunk_data['_complete_content'] = complete_content
                
                if is_sse:
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n".encode('utf-8')
                else:
                    yield json.dumps(chunk_data, ensure_ascii=False).encode('utf-8') + b'\n'
            else:
                # 字符串数据
                if is_sse:
                    yield f"data: {chunk_data}\n\n".encode('utf-8')
                else:
                    yield chunk_data.encode('utf-8') + b'\n'
        
        # 确保发送 [DONE] 标记
        yield b"data: [DONE]\n\n"

async def proxy_streaming_request(request: Request, endpoint: str) -> StreamingResponse:
    """代理流式请求，处理 chat/completions 和 completions 接口"""
    # 生成请求ID用于日志跟踪
    request_id = str(uuid.uuid4())
    
    # 读取请求体用于日志记录
    request_body = await request.body()
    user_input = ""
    messages = None
    prompt = None
    is_streaming = False
    try:
        request_data = json.loads(request_body)
        # 检查是否是流式请求
        is_streaming = request_data.get("stream", False)
        
        if endpoint == "/v1/chat/completions" and "messages" in request_data:
            # 获取最后一条用户消息
            messages = request_data.get("messages", [])
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_input = msg.get("content", "")
                    break
        elif endpoint == "/v1/completions" and "prompt" in request_data:
            prompt = request_data.get("prompt", "")
            user_input = prompt
    except:
        user_input = "无法解析请求内容"
    
    async with httpx.AsyncClient(timeout=60.0*10) as client:
        response = await client.request(
            method=request.method,
            url=f"{VLLM_API_URL}{endpoint}",
            headers={k: v for k, v in request.headers.items() if k.lower() != 'host'},
            content=request_body,
        )
        
        # 创建一个包装的流生成器来收集AI回复
        async def wrapped_stream():
            ai_response = ""
            async for chunk in process_stream(response, endpoint, request_id):
                # 尝试解析chunk以收集AI回复
                try:
                    chunk_str = chunk.decode('utf-8').strip()
                    if chunk_str:
                        # 处理 SSE 格式
                        if chunk_str.startswith('data: '):
                            json_str = chunk_str[6:]  # 移除 'data: ' 前缀
                            if json_str != '[DONE]':
                                try:
                                    chunk_data = json.loads(json_str)
                                    # 检查是否有最终回复标记
                                    if '_final_response' in chunk_data and '_request_id' in chunk_data:
                                        ai_response = chunk_data['_final_response']
                                        # 异步记录日志，不阻塞响应
                                        asyncio.create_task(logger.log_interaction(
                                            request_id, user_input, ai_response, endpoint, messages, prompt, is_streaming
                                        ))
                                        # 清理临时字段
                                        del chunk_data['_final_response']
                                        del chunk_data['_request_id']
                                        if '_complete_content' in chunk_data:
                                            del chunk_data['_complete_content']
                                        chunk = f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n".encode('utf-8')
                                except json.JSONDecodeError:
                                    pass
                        else:
                            # 处理非 SSE 格式
                            try:
                                chunk_data = json.loads(chunk_str)
                                # 检查是否有最终回复标记
                                if '_final_response' in chunk_data and '_request_id' in chunk_data:
                                    ai_response = chunk_data['_final_response']
                                    # 异步记录日志，不阻塞响应
                                    asyncio.create_task(logger.log_interaction(
                                        request_id, user_input, ai_response, endpoint, messages, prompt, is_streaming
                                    ))
                                    # 清理临时字段
                                    del chunk_data['_final_response']
                                    del chunk_data['_request_id']
                                    if '_complete_content' in chunk_data:
                                        del chunk_data['_complete_content']
                                    chunk = json.dumps(chunk_data, ensure_ascii=False).encode('utf-8') + b'\n'
                            except json.JSONDecodeError:
                                pass
                except:
                    pass
                yield chunk
        
        return StreamingResponse(
            content=wrapped_stream(),
            status_code=response.status_code,
            headers={k: v for k, v in response.headers.items() if k.lower() not in ('transfer-encoding', 'content-length')},
            media_type="application/json"
        )

async def proxy_direct_request(request: Request, endpoint: str) -> Response:
    """直接代理其他接口，保持原样返回"""
    # 生成请求ID用于日志跟踪
    request_id = str(uuid.uuid4())
    
    # 读取请求体用于日志记录
    request_body = await request.body()
    user_input = ""
    messages = None
    prompt = None
    is_streaming = False
    
    # 只对chat和completions接口记录日志
    should_log = endpoint in ["/v1/chat/completions", "/v1/completions"]
    
    if should_log:
        try:
            request_data = json.loads(request_body)
            # 检查是否是流式请求
            is_streaming = request_data.get("stream", False)
            
            if endpoint == "/v1/chat/completions" and "messages" in request_data:
                # 获取最后一条用户消息
                messages = request_data.get("messages", [])
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        user_input = msg.get("content", "")
                        break
            elif endpoint == "/v1/completions" and "prompt" in request_data:
                prompt = request_data.get("prompt", "")
                user_input = prompt
        except:
            user_input = "无法解析请求内容"
    
    async with httpx.AsyncClient(timeout=60.0*10) as client:
        response = await client.request(
            method=request.method,
            url=f"{VLLM_API_URL}{endpoint}",
            headers={k: v for k, v in request.headers.items() if k.lower() != 'host'},
            content=request_body,
        )
        
        # 对于非流式的chat和completions请求，记录AI回复
        if should_log and not is_streaming:
            try:
                print(f"[DEBUG] Processing non-streaming response for {endpoint}")
                print(f"[DEBUG] Response status: {response.status_code}")
                print(f"[DEBUG] Response content length: {len(response.content)}")
                print(f"[DEBUG] Response content preview: {response.content[:500]}")
                
                response_data = json.loads(response.content)
                ai_response = ""
                
                if endpoint == "/v1/chat/completions" and response_data.get('choices'):
                    ai_response = response_data['choices'][0].get('message', {}).get('content', '')
                elif endpoint == "/v1/completions" and response_data.get('choices'):
                    ai_response = response_data['choices'][0].get('text', '')
                
                print(f"[DEBUG] Extracted AI response: {repr(ai_response)}")
                
                # 检查是否包含 ### Direct Answer:\n 并处理
                if ai_response and '### Direct Answer:\n' in ai_response:
                    print(f"[DEBUG] Detected Direct Answer in non-streaming response: {repr(ai_response)}")
                    parts = ai_response.split('### Direct Answer:\n', 1)
                    if len(parts) > 1:
                        direct_answer_content = parts[1].lstrip()
                        print(f"[DEBUG] Extracted direct answer: {repr(direct_answer_content)}")
                        
                        # 修改响应内容，只返回 Direct Answer 后的部分
                        if endpoint == "/v1/chat/completions":
                            response_data['choices'][0]['message']['content'] = direct_answer_content
                        elif endpoint == "/v1/completions":
                            response_data['choices'][0]['text'] = direct_answer_content
                        
                        # 更新响应内容
                        response_content = json.dumps(response_data, ensure_ascii=False).encode('utf-8')
                        
                        ai_response = direct_answer_content
                        
                        # 异步记录日志，不阻塞响应
                        asyncio.create_task(logger.log_interaction(
                            request_id, user_input, ai_response, endpoint, messages, prompt, is_streaming
                        ))
                        
                        print(f"[DEBUG] Returning modified response with direct answer")
                        # 返回修改后的响应
                        return Response(
                            content=response_content,
                            status_code=response.status_code,
                            headers={k: v for k, v in response.headers.items() if k.lower() not in ('transfer-encoding', 'content-length')},
                            media_type=response.headers.get('content-type', 'application/json')
                        )
                else:
                    print(f"[DEBUG] No Direct Answer found, returning original response")
                
                # 正常情况下异步记录日志，不阻塞响应
                asyncio.create_task(logger.log_interaction(
                    request_id, user_input, ai_response, endpoint, messages, prompt, is_streaming
                ))
            except Exception as e:
                print(f"[ERROR] Error processing response for {endpoint}: {e}")
                import traceback
                traceback.print_exc()
        
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers={k: v for k, v in response.headers.items() if k.lower() != 'transfer-encoding'},
            media_type=response.headers.get('content-type')
        )

# 智能路由：根据请求类型选择处理方式
@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    # 预读请求体来判断是否为流式请求
    request_body = await request.body()
    try:
        request_data = json.loads(request_body)
        is_streaming = request_data.get("stream", False)
    except:
        is_streaming = False
    
    # 重新创建请求对象
    async def new_receive():
        return {"type": "http.request", "body": request_body}
    
    new_request = Request(scope=request.scope, receive=new_receive)
    
    if is_streaming:
        return await proxy_streaming_request(new_request, "/v1/chat/completions")
    else:
        return await proxy_direct_request(new_request, "/v1/chat/completions")

@app.post("/v1/completions")
async def proxy_completions(request: Request):
    # 预读请求体来判断是否为流式请求
    request_body = await request.body()
    try:
        request_data = json.loads(request_body)
        is_streaming = request_data.get("stream", False)
    except:
        is_streaming = False
    
    # 重新创建请求对象
    async def new_receive():
        return {"type": "http.request", "body": request_body}
    
    new_request = Request(scope=request.scope, receive=new_receive)
    
    if is_streaming:
        return await proxy_streaming_request(new_request, "/v1/completions")
    else:
        return await proxy_direct_request(new_request, "/v1/completions")

# 通用路由：直接透传其他接口
@app.api_route("/{path:path}", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def catch_all(request: Request, path: str):
    return await proxy_direct_request(request, f"/{path}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)