# 线上服务
## 路径

/maindata/data/shared/public/online
其中的东西，谨慎操作

## 更新文件

cp /maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/del_paln/vlm_ser_claude.py /maindata/data/shared/public/online/server



## 启动sglang 服务

### mistral
python3 -m sglang.launch_server \
  --model-path /maindata/data/shared/public/wanpenghan/LLaMA-Factory/saves/full/mistral-24B-ins-sft_ML_6w_0609_lr3e6/checkpoint-21070 \
  --served-model-name MindLink_Beta \
  --host 0.0.0.0 \
  --port 3280 \
  --chat-template /maindata/data/shared/public/online/chat_template/chat_template_plan.jinja \
  --tp 2 \
  --context-length 32768 \
  --max-prefill-tokens 100000 \
  --max-running-requests 300 \
  --mem-fraction-static 0.95 \
  --trust-remote-code



### qwen
python3 -m sglang.launch_server \
  --model-path /maindata/data/shared/public/wanpenghan/LLaMA-Factory/saves/full/Qwen3-32B-ins-sft_ML_2w_0612_lr1e5/checkpoint-7830 \
  --served-model-name MindLink_Beta \
  --host 0.0.0.0 \
  --port 3280 \
  --chat-template /maindata/data/shared/public/online/chat_template/qwen3_plan.jinja \
  --tp 2 \
  --context-length 32768 \
  --max-prefill-tokens 100000 \
  --max-running-requests 300 \
  --mem-fraction-static 0.95 \
  --trust-remote-code

## 部署web 服务

python3 /maindata/data/shared/public/online/server/vlm_ser_claude.py

## 请求web_log 

curl -X POST https://sd15vu0fhj5i8uvr669og.apigateway-cn-beijing.volceapi.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 5d18423d-d119-489c-a780-8a76924228d2" \
  -d '{
    "model": "MindLink_Beta",
    "messages": [
      {
        "role": "system",
        "content": "你是个AI助手"
      },
      {
        "role": "user",
        "content": "你好，这个接口能正常工作吗？"
      }
    ],
    "max_tokens": 1000,
    "stream": false
}'

### 请求LLM


curl -X POST https://sd0kkieqcirbt02vttd60.apigateway-cn-beijing.volceapi.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 5d18423d-d119-489c-a780-8a76924228d2" \
  -d '{
    "model": "MindLink_Beta",
    "messages": [
      {"role": "user", "content": "你是谁"}
    ],
    "max_tokens": 2000,
    "stream": true
}'


