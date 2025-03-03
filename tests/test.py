from transformers import AutoTokenizer

model_path = "/maindata/data/shared/public/yangchao.zhou/models/mistralai/Mistral-Nemo-Base-2407"

tokenizer = AutoTokenizer.from_pretrained(model_path)

# 确保 chat_template 被正确加载
print(tokenizer.chat_template)

new_model_path = "/maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/tests"
tokenizer.save_pretrained(model_path)
print(f"模型已保存到 {new_model_path}。")