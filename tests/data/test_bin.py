import os

local_path = '/maindata/data/shared/public/yangchao.zhou/projects/mistral_pro/data/instruction/open_source/generated_sft_data/AIME_2025_processed_data_Llama-33-70b-ins-sft_qwen_gpt_20250406_2give_ans.bin'
T = os.path.isfile(local_path)
print(T)