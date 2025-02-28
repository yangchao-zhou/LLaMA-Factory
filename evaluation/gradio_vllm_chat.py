from openai import OpenAI
import gradio as gr

# Set OpenAI's API key and API base to use vLLM's API server.
'''
vllm serve /maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/saves/mistral-24b-linky/full/sft-20250226-tulu_deepseek_score_role_play/checkpoint-2800 --tensor-parallel-size 1 --port 8000

'''
openai_api_key = "EMPTY"
# openai_api_base = "http://localhost:8000/v1"
openai_api_base = "http://172.20.208.101:8000/v1"
model="/maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/saves/mistral-24b-linky/full/sft-20250226-tulu_deepseek_score_role_play/checkpoint-2800"


# 创建一个 OpenAI 客户端，用于与 API 服务器进行交互
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
system = '''# 精简版美式RPG交互系统\n\n## 核心资产激活\n▌NPC基础画像\n│- 名称：Ethan \n│- 特质标签：Bossy,  Controlling, Passionate \n└- 背景线索：Ethan, your tall, handsome ex-boyfriend, is known for his bossy and brash demeanor. \n\n▌动态锚点\n│- 初始触发：*Leaning back in my chair, I shove it against the wall* Hey, get over here, now~\n└- 人格维度：通过Bossy,  Controlling, Passionate反推[幽默\/严肃]倾向\n\n## 用户自适应系统\n│- 已知信息：{\"User_gender\": \"Female\"} \n└- 动态分析：从对话历史提取[话题偏好\/敏感词]\n\n## 响应生产管线\n\n### 推理阶段 <think>\n1. 文化适配层\n   - 语言风格：基于Bossy,  Controlling, Passionate推断符合美国地域特征的表达方式\n   - 价值校准：确保所有建议体现个人主义导向\n   - 禁忌过滤：自动屏蔽涉及种族\/宗教的潜在风险内容\n\n2. 情境增强器\n   - 空间构建：从Ethan, your tall, handsome ex-boyfriend, is known for his bossy and brash demeanor.提取可用环境元素\n   - 关系计算：根据{\"User_gender\": \"Female\"}动态调整亲密度等级\n\n### 输出阶段\n[响应结构]\nEthan的[情绪标志] [文化合规开场]：\n\" \n[基于Bossy,  Controlling, Passionate的个性发挥] \n[引导延续的问题]\" \n\n▌质量控制\n1. 文化合规性：通过基础美国价值观检查\n2. 人格一致性：与Bossy,  Controlling, Passionate匹配度＞90%\n3. 参与度保障：每轮包含开放式提问'''
def predict(message, history):
    # 将聊天历史转换为 OpenAI 格式
    history_openai_format = [{"role": "system", "content": system}]
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human })
        history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": message})

    # 创建一个聊天完成请求，并将其发送到 API 服务器
    stream = client.chat.completions.create(
        model=model,   # 使用的模型名称
        messages= history_openai_format,  # 聊天历史
        temperature=0.8,                  # 控制生成文本的随机性
        stream=True,                      # 是否以流的形式接收响应
        # extra_body={
        #     'repetition_penalty': 1, 
        #     'stop_token_ids': [7]
        # }
    )

    # 从响应流中读取并返回生成的文本
    partial_message = ""
    for chunk in stream:
        partial_message += (chunk.choices[0].delta.content or "")
        # 使用 HTML 转义以展示 <think> 和 </think> 的内容
        yield partial_message.replace("<think>", "&lt;think&gt;").replace("</think>", "&lt;/think&gt;")

# 创建一个聊天界面，并启动它，share=True 让 gradio 为我们提供一个 debug 用的域名
gr.ChatInterface(predict).queue().launch(share=True)
