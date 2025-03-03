from openai import OpenAI
import gradio as gr

# Set OpenAI's API key and API base to use vLLM's API server.
'''
sudo -s
cd /maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory
conda activate nemo
vllm serve /maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/saves/mistral-24b-linky/full/bak/multi_turn/checkpoint-1800 --tensor-parallel-size 1 --port 8000
ps aux | grep vllm | grep -v grep | awk '{print $2}' | xargs kill -9

'''
openai_api_key = "EMPTY"
# openai_api_base = "http://localhost:8000/v1"
openai_api_base = "http://localhost:8000/v1"
model="/maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/saves/mistral-24b-linky/full/bak/multi_turn/checkpoint-1800"


# 创建一个 OpenAI 客户端，用于与 API 服务器进行交互
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# 默认 system 提示已移入前端配置，不再硬编码

def predict(message, history, frontend_system, frontend_temperature, max_tokens, top_p):
    # 使用前端传入的 system 提示
    history_openai_format = [{"role": "system", "content": frontend_system}]
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human })
        history_openai_format.append({"role": "assistant", "content": assistant})
    history_openai_format.append({"role": "user", "content": message.replace("&lt;think&gt;", "<think>").replace("&lt;/think&gt;", "</think>")})
    stream = client.chat.completions.create(
        model=model,
        messages=history_openai_format,
        temperature=frontend_temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=True,
        reasoning_effort="low",
    )
    partial_message = ""
    for chunk in stream:
        partial_message += (chunk.choices[0].delta.content or "")
        # print("Partial message:", partial_message)
        yield partial_message.replace("<think>", "&lt;think&gt;").replace("</think>", "&lt;/think&gt;")
        # yield partial_message


def user(message, chat_history, sys_prompt, temp, max_tokens, top_p):
    chat_history = chat_history or []
    # 保留之前的历史记录，只处理新消息
    history_length = len(chat_history)
    
    for partial in predict(message, chat_history, sys_prompt, temp, max_tokens, top_p):
        # 如果是第一次添加新消息
        if len(chat_history) == history_length:
            chat_history.append((message, partial))
        else:
            # 更新最后一条消息的回复部分
            chat_history[-1] = (message, partial)
        yield chat_history, chat_history

def clear_history():
    return [], []  # 清空 chatbot 和 state

with gr.Blocks() as demo:
    with gr.Row():
        system_prompt = gr.Textbox(label="System Prompt", value="Hello! I'm an AI assistant that can help you with a variety of tasks.")
        temperature_slider = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
        max_tokens_slider = gr.Slider(label="max_tokens", minimum=1, maximum=32000, value=8192, step=1)
        top_p_slider = gr.Slider(label="Top P", minimum=0.0, maximum=1.0, value=0.9, step=0.01)
    chatbot = gr.Chatbot()
    with gr.Row():
        msg = gr.Textbox(label="Your Message")
        clear_btn = gr.Button("🗑️ Clear History")  # 添加清理按钮
    state = gr.State([])

    # 绑定清理按钮事件
    clear_btn.click(
        fn=clear_history,
        outputs=[chatbot, state],
        queue=False
    )

    # 注意：提交触发后更新聊天记录，gr.Blocks 可自动处理流式生成
    msg.submit(
        user,
        inputs=[msg, state, system_prompt, temperature_slider, max_tokens_slider, top_p_slider],
        outputs=[chatbot, state],
        queue=True  # 确保消息按顺序处理
    ).then(
        lambda: "",  # 清空输入框
        None,
        msg
    )
    demo.queue()  # 启用队列以支持流式处理
    demo.launch(share=True)
