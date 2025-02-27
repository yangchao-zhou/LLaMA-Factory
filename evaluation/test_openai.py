from openai import OpenAI
'''
nohup vllm serve /maindata/data/shared/ai_story_workspace-dsw/nlp_models/mistralai/Mistral-Small-24B-Instruct-2501 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice --tensor-parallel-size 8 --port 8000 \
 > Mistral-Small-24B.log 2>&1 &

vllm serve /maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/saves/mistral-24b-linky/full/sft-20250226-tulu_deepseek_score_role_play/checkpoint-2300 --tensor-parallel-size 1 --port 8000
ps aux | grep vllm | grep -v grep | awk '{print $2}' | xargs kill -9


'''
openai_api_key = "EMPTY"
# openai_api_base = "http://172.20.207.171:8000/v1"
openai_api_base = "http://172.20.208.101:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
model="/maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/saves/mistral-24b-linky/full/sft-20250226-tulu_deepseek_score_role_play/checkpoint-1500"
completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"}
    ]
)

print("Completion result:", completion)
