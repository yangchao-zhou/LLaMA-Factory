from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import json
import logging
import asyncio
import time # For created timestamp fallback
from typing import Dict, Any # For type hinting

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Upstream VLLM service URL (from the curl command)
UPSTREAM_URL = "https://sd0kkieqcirbt02vttd60.apigateway-cn-beijing.volceapi.com"
DIRECT_ANSWER_MARKER = "### Direct Answer:"

async def process_and_filter_content(content: str) -> str:
    marker_pos = content.find(DIRECT_ANSWER_MARKER)
    if marker_pos != -1:
        new_content = content[marker_pos + len(DIRECT_ANSWER_MARKER):].lstrip()
        return new_content
    return content

async def forward_request(request: Request, endpoint: str):
    client_headers = dict(request.headers)
    headers_to_forward = {
        "Content-Type": client_headers.get("content-type", "application/json"),
    }
    if "authorization" in client_headers:
        headers_to_forward["Authorization"] = client_headers["authorization"]

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        logger.error("Invalid JSON payload received.")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    is_streaming = payload.get("stream", False)
    target_url = f"{UPSTREAM_URL}{endpoint}"

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            if not is_streaming:
                logger.info(f"Forwarding non-streaming request to {target_url}")
                response = await client.post(target_url, json=payload, headers=headers_to_forward)
                
                response_headers = dict(response.headers)
                filtered_response_headers = {
                    k: v for k, v in response_headers.items() 
                    if k.lower() not in ['content-length', 'content-type', 'transfer-encoding', 'connection']
                }

                if response.status_code >= 400:
                    try:
                        error_detail = response.json()
                    except json.JSONDecodeError:
                        error_detail = response.text
                    logger.error(f"Upstream error for non-streaming: {response.status_code} - {error_detail}")
                    return JSONResponse(content=error_detail, status_code=response.status_code, headers=filtered_response_headers)

                response_json = response.json()
                
                if endpoint == "/v1/chat/completions":
                    if response_json.get("choices") and len(response_json["choices"]) > 0 and response_json["choices"][0].get("message"):
                        original_content = response_json["choices"][0]["message"].get("content", "")
                        filtered_content = await process_and_filter_content(original_content)
                        response_json["choices"][0]["message"]["content"] = filtered_content
                elif endpoint == "/v1/completions":
                    if response_json.get("choices") and len(response_json["choices"]) > 0 and response_json["choices"][0].get("text") is not None:
                        original_content = response_json["choices"][0].get("text", "")
                        filtered_content = await process_and_filter_content(original_content)
                        response_json["choices"][0]["text"] = filtered_content
                
                return JSONResponse(content=response_json, headers=filtered_response_headers, status_code=response.status_code)

            else: # Streaming
                logger.info(f"Forwarding streaming request to {target_url}")
                
                _upstream_headers: Dict[str, str] = {}
                _upstream_status_code: int = 200 
                _original_event_lines: list[str] = []

                async def stream_generator():
                    nonlocal _upstream_headers, _upstream_status_code, _original_event_lines
                    
                    accumulated_content_parts = []
                    is_chat_endpoint = endpoint == "/v1/chat/completions"
                    first_event_data_template: Dict[str, Any] | None = None
                    final_finish_reason: str | None = None

                    try:
                        async with client.stream("POST", target_url, json=payload, headers=headers_to_forward) as upstream_response:
                            _upstream_status_code = upstream_response.status_code
                            _upstream_headers.update(dict(upstream_response.headers))

                            if _upstream_status_code != 200:
                                error_content_bytes = await upstream_response.aread()
                                error_content = error_content_bytes.decode('utf-8', errors='ignore')
                                logger.error(f"Upstream error at stream start: {_upstream_status_code} - {error_content}")
                                yield f"data: {json.dumps({'error': 'Upstream request failed at start', 'status_code': _upstream_status_code, 'detail': error_content})}\n\n"
                                yield "data: [DONE]\n\n"
                                return

                            buffer = ""
                            async for chunk in upstream_response.aiter_bytes():
                                buffer += chunk.decode('utf-8', errors='ignore')
                                
                                while '\n' in buffer:
                                    line, buffer = buffer.split('\n', 1)
                                    line = line.strip() 
                                    if not line: 
                                        continue
                                    
                                    _original_event_lines.append(line)

                                    if line.startswith("data: "):
                                        json_str = line[6:].strip()
                                        if json_str == "[DONE]":
                                            if final_finish_reason is None: 
                                                final_finish_reason = "stop" 
                                            continue 
                                        try:
                                            data: Dict[str, Any] = json.loads(json_str)
                                            if first_event_data_template is None and data:
                                                first_event_data_template = data 

                                            content_part = ""
                                            if data.get("choices") and len(data["choices"]) > 0:
                                                choice = data["choices"][0]
                                                if choice.get("finish_reason"):
                                                    final_finish_reason = choice.get("finish_reason")
                                                if is_chat_endpoint:
                                                    if choice.get("delta"):
                                                        content_part = choice["delta"].get("content", "")
                                                else: 
                                                    content_part = choice.get("text", "")
                                            if content_part:
                                                accumulated_content_parts.append(content_part)
                                        except json.JSONDecodeError:
                                            logger.warning(f"Could not parse JSON from stream: {json_str}")
                                        except Exception as e:
                                            logger.warning(f"Error processing stream data line: {line}, error: {e}")
                            
                            if buffer.strip(): 
                                line = buffer.strip()
                                _original_event_lines.append(line)
                                if line.startswith("data: "):
                                    json_str = line[6:].strip()
                                    if json_str != "[DONE]":
                                        try:
                                            data = json.loads(json_str)
                                            if first_event_data_template is None and data:
                                                first_event_data_template = data
                                            content_part = ""
                                            if data.get("choices") and len(data["choices"]) > 0:
                                                choice = data["choices"][0]
                                                if choice.get("finish_reason"):
                                                    final_finish_reason = choice.get("finish_reason")
                                                if is_chat_endpoint:
                                                    if choice.get("delta"):
                                                        content_part = choice["delta"].get("content", "")
                                                else:
                                                    content_part = choice.get("text", "")
                                            if content_part:
                                                accumulated_content_parts.append(content_part)
                                        except Exception: 
                                            pass
                                    elif final_finish_reason is None: 
                                        final_finish_reason = "stop"

                    
                    except httpx.HTTPStatusError as e:
                        logger.error(f"HTTPStatusError from upstream during stream setup: {e.response.status_code} - {e.response.text}")
                        _upstream_status_code = e.response.status_code 
                        yield f"data: {json.dumps({'error': 'Upstream request failed', 'status_code': e.response.status_code, 'detail': e.response.text})}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    except httpx.HTTPError as e:
                        logger.error(f"HTTPError during streaming from upstream: {e}")
                        _upstream_status_code = 503 
                        yield f"data: {json.dumps({'error': 'Upstream connection error', 'detail': str(e)})}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    except Exception as e:
                        logger.error(f"Unexpected error during upstream call: {e}", exc_info=True)
                        _upstream_status_code = 500 
                        yield f"data: {json.dumps({'error': 'Unexpected error during proxying', 'detail': str(e)})}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    full_text_content = "".join(accumulated_content_parts)
                    filtered_text_content = await process_and_filter_content(full_text_content)

                    if full_text_content != filtered_text_content:
                        logger.info("Content filtered. Re-streaming modified content.")
                        
                        base_event: Dict[str, Any] = {}
                        model_name_to_use = payload.get("model", "unknown-model")
                        if first_event_data_template and first_event_data_template.get("model"):
                            model_name_to_use = first_event_data_template.get("model")

                        if first_event_data_template:
                             base_event = {
                                "id": first_event_data_template.get("id", f"chatcmpl-filtered-{int(time.time())}"),
                                "object": first_event_data_template.get("object", "chat.completion.chunk" if is_chat_endpoint else "text_completion"),
                                "created": first_event_data_template.get("created", int(time.time())),
                                "model": model_name_to_use,
                            }
                             system_fingerprint = first_event_data_template.get("system_fingerprint")
                             if system_fingerprint is not None:
                                 base_event["system_fingerprint"] = system_fingerprint
                        else: 
                             base_event = {
                                "id": f"chatcmpl-filtered-{int(time.time())}",
                                "object": "chat.completion.chunk" if is_chat_endpoint else "text_completion",
                                "created": int(time.time()),
                                "model": model_name_to_use,
                            }
                        
                        content_choice_data: Dict[str, Any] = {"index": 0}
                        if is_chat_endpoint:
                            content_choice_data["delta"] = {"content": filtered_text_content}
                        else: 
                            content_choice_data["text"] = filtered_text_content
                        
                        event_with_content = {**base_event, "choices": [content_choice_data]}
                        yield f"data: {json.dumps(event_with_content)}\n\n"
                        
                        finish_choice_data: Dict[str, Any] = {"index": 0, "finish_reason": final_finish_reason if final_finish_reason else "stop"}
                        if is_chat_endpoint:
                            finish_choice_data["delta"] = {} 
                        
                        event_with_finish = {**base_event, "choices": [finish_choice_data]}
                        yield f"data: {json.dumps(event_with_finish)}\n\n"
                        yield "data: [DONE]\n\n"
                    else:
                        logger.info("No filtering needed. Replaying original stream.")
                        has_done_event = False
                        for line_content in _original_event_lines:
                            yield f"{line_content}\n" 
                            if line_content.strip() == "data: [DONE]":
                                has_done_event = True
                        if not has_done_event: 
                             yield "data: [DONE]\n\n"
                
                gen_obj = stream_generator()
                return StreamingResponse(gen_obj, media_type=_upstream_headers.get("content-type", "text/event-stream"), status_code=_upstream_status_code)

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTPStatusError from upstream for {target_url}: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except httpx.RequestError as e: 
            logger.error(f"RequestError for {target_url}: {e}")
            raise HTTPException(status_code=503, detail=f"Upstream service request error: {str(e)}")
        except Exception as e: 
            logger.error(f"Unexpected error proxying to {target_url}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    return await forward_request(request, "/v1/chat/completions")

@app.post("/v1/completions")
async def proxy_completions(request: Request):
    return await forward_request(request, "/v1/completions")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting VLLM proxy server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
