import wandb
import json

# 初始化 wandb
wandb.init(project="llamafactory", entity="combined_sft-20250226-tulu_deepseek_score_role_play")

# 读取 trainer_log.jsonl 文件
log_file_path = '/maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/saves/mistral-24b-linky/full/sft-20250226-tulu_deepseek_score_role_play/trainer_log.jsonl'

# 打开文件并上传每一行数据到 wandb
with open(log_file_path, 'r') as file:
    for line in file:
        log_entry = json.loads(line.strip())  # 每一行是一个 JSON 对象
        # 上传日志数据
        wandb.log(log_entry)

# 完成
wandb.finish()