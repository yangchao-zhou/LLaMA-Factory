import httpx
import json
import asyncio
import logging
from typing import List, Dict

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 代理服务地址
PROXY_API_URL = "http://localhost:8000"

async def get_models() -> List[Dict]:
    """获取 vLLM 的模型列表"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{PROXY_API_URL}/v1/models")
            response.raise_for_status()
            models = response.json().get('data', [])
            logger.info(f"Retrieved models: {[model['id'] for model in models]}")
            return models
        except httpx.HTTPError as e:
            logger.error(f"Failed to get models: {e}")
            return []

async def test_streaming_request(endpoint: str, model: str, prompt: str, stream: bool = True) -> None:
    """测试流式或非流式请求"""
    payload = {
        "model": model,
        "prompt": prompt if endpoint == "/v1/completions" else None,
        "messages": [{"role": "user", "content": prompt}] if endpoint == "/v1/chat/completions" else None,
        "stream": stream,
        "max_tokens": 10000,  # 添加 max_tokens 参数
        "temperature": 0.7  # 可选：添加温度参数
    }
    payload = {k: v for k, v in payload.items() if v is not None}  # 移除空字段
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            if stream:
                logger.info(f"Testing streaming request to {endpoint} with model {model}")
                async with client.stream("POST", f"{PROXY_API_URL}{endpoint}", json=payload) as response:
                    response.raise_for_status()
                    print(f"\nStreaming response for {endpoint} (model: {model}):")
                    async for chunk in response.aiter_bytes():
                        print(chunk.decode('utf-8', errors='ignore'), end='', flush=True)
                    print("\n")
            else:
                logger.info(f"Testing non-streaming request to {endpoint} with model {model}")
                response = await client.post(f"{PROXY_API_URL}{endpoint}", json=payload)
                response.raise_for_status()
                print(f"\nNon-streaming response for {endpoint} (model: {model}):")
                print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        except httpx.HTTPError as e:
            logger.error(f"Request to {endpoint} failed: {e}")

async def main():
    # 测试用例的 prompt，包含 ### Direct Answer
    test_prompt = '''Yarik is a big fan of many kinds of music. But Yarik loves not only listening to music but also writing it. He likes electronic music most of all, so he has created his own system of music notes, which, in his opinion, is best for it.\n\nSince Yarik also likes informatics, in his system notes are denoted by integers of $2^k$, where $k \\ge 1$ — a positive integer. But, as you know, you can't use just notes to write music, so Yarik uses combinations of two notes. The combination of two notes $(a, b)$, where $a = 2^k$ and $b = 2^l$, he denotes by the integer $a^b$.\n\nFor example, if $a = 8 = 2^3$, $b = 4 = 2^2$, then the combination $(a, b)$ is denoted by the integer $a^b = 8^4 = 4096$. Note that different combinations can have the same notation, e.g., the combination $(64, 2)$ is also denoted by the integer $4096 = 64^2$.\n\nYarik has already chosen $n$ notes that he wants to use in his new melody. However, since their integers can be very large, he has written them down as an array $a$ of length $n$, then the note $i$ is $b_i = 2^{a_i}$. The integers in array $a$ can be repeated.\n\nThe melody will consist of several combinations of two notes. Yarik was wondering how many pairs of notes $b_i, b_j$ $(i < j)$ exist such that the combination $(b_i, b_j)$ is equal to the combination $(b_j, b_i)$. In other words, he wants to count the number of pairs $(i, j)$ $(i < j)$ such that $b_i^{b_j} = b_j^{b_i}$. Help him find the number of such pairs.\n\nInput\n\nThe first line of the input contains one integer $t$ ($1 \\le t \\le 10^4$) — the number of test cases.\n\nThe first line of each test case contains one integer $n$ ($1 \\leq n \\leq 2 \\cdot 10^5$) — the length of the arrays.\n\nThe next line contains $n$ integers $a_1, a_2, \\dots, a_n$ ($1 \\leq a_i \\leq 10^9$) — array $a$.\n\nIt is guaranteed that the sum of $n$ over all test cases does not exceed $2 \\cdot 10^5$.\n\nOutput\n\nFor each test case, output the number of pairs that satisfy the given condition.Sample Input 1:\n5\n\n1\n\n2\n\n4\n\n3 1 3 2\n\n2\n\n1000 1000\n\n3\n\n1 1 1\n\n19\n\n2 4 1 6 2 8 5 4 2 10 5 10 8 7 4 3 2 6 10\n\n\n\nSample Output 1:\n\n0\n2\n1\n3\n19\n"
    '''

    # test_prompt = "我爱你，但是"

    # 获取模型列表
    models = await get_models()
    if not models:
        logger.error("No models available. Exiting.")
        return

    # 选择第一个模型进行测试
    model = models[0]['id']
    logger.info(f"Using model: {model}")

    # 测试两个接口：流式和非流式
    endpoints = ["/v1/chat/completions", "/v1/completions"]
    for endpoint in endpoints:
        print(f"\nTesting endpoint: {endpoint}")
        # 测试流式请求
        await test_streaming_request(endpoint, model, test_prompt, stream=True)
        # 测试非流式请求
        await test_streaming_request(endpoint, model, test_prompt, stream=False)

        # 等待 1 秒，避免请求过快
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())