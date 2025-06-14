## 请求web_log 

curl -X POST https://sd15vu0fhj5i8uvr669og.apigateway-cn-beijing.volceapi.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 5d18423d-d119-489c-a780-8a76924228d2" \
  -d '{
    "model": "ML-ep13",
    "messages": [
      {
        "role": "user",
        "content": "你好，这个接口能正常工作吗？"
      }
    ],
    "max_tokens": 1000,
    "stream": true
}'

### 请求LLM


curl -X POST https://sd0kkieqcirbt02vttd60.apigateway-cn-beijing.volceapi.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 5d18423d-d119-489c-a780-8a76924228d2" \
  -d '{
    "model": "ML-ep13",
    "messages": [
      {"role": "user", "content": "你是谁"}
    ],
    "max_tokens": 2000,
    "stream": true
}'
