## todo

尝试冻结一部分层
可以试试lora 的效果

合并后的数据数量：54579

基于长COT的角色扮演(生成针对回答的推理过程，然后一起训练)
把有comment的打分数据参与训练
试试之前指令模型开始训练的效果

生成聊天数据的数据
如何让history 不参与训练
请用长思维链的方式回答。回答之前先思考一下，长思维链的内容在<think>和</think>之间。思维链之外的内容才是真的回答

测一下基础模型的能力，引入长思考链到模型的推理过程中。(一坨屎)

## 训练

pip install -e ".[torch,metrics]"
pip install deepspeed==0.14.5
pip install flash-attn==2.6.2
pip install lmdeploy
export  PYTHONPATH=`pwd`

export TMPDIR=/maindata/data/shared/public/yangchao.zhou/projects/tmp

conda activate nemo
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
llamafactory-cli train examples/train_full/mistral_full_sft_ds.yaml
nohup llamafactory-cli train examples/train_full/mistral_full_sft_ds.yaml > train_output.log 2>&1 &

sudo chown -R ran.xiao /maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory
sudo chmod -R 777 /maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/
sudo chmod -R 777 /maindata/data/shared/public/yangchao.zhou/projects/spanish

## 部署网页版

conda activate nemo

export CUDA_VISIBLE_DEVICES=0

lmdeploy serve gradio /maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/saves/mistral-24b-linky/full/sft-20250226-tulu_deepseek_score_role_play/checkpoint-1500
lmdeploy serve gradio /maindata/data/shared/ai_story_workspace-dsw/nlp_models/mistralai/Mistral-Small-24B-Instruct-2501
lmdeploy serve gradio /maindata/data/shared/public/yangchao.zhou/models/mistralai/Mistral-Small-24B-Base-2501

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
lmdeploy serve gradio /maindata/data/shared/ai_story_workspace-dsw/nlp_models/mistralai/Mistral-Small-24B-Instruct-2501 --tp 8

## vllm 部署
nohup vllm serve /maindata/data/shared/ai_story_workspace-dsw/nlp_models/mistralai/Mistral-Small-24B-Instruct-2501 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice --tensor-parallel-size 8 --port 8000 \
 > Mistral-Small-24B.log 2>&1 &
 
vllm serve /maindata/data/shared/ai_story_workspace-dsw/nlp_models/mistralai/Mistral-Small-24B-Instruct-2501 --tensor-parallel-size 8 --port 8000

nohup python -m vllm.entrypoints.openai.api_server \
    --model /maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/saves/mistral-24b-linky/full/sft-20250226-tulu_deepseek_score_role_play/checkpoint-1500\
    --tensor-parallel-size 8 \
    --disable-custom-all-reduce \
    --trust-remote-code \
    --port 8000 > Mistral-Small-24B.log 2>&1 &

## eval

conda activate nemo
export  PYTHONPATH=`pwd`
python src/eval_es_tmp-200.py

conda activate nemo
export CUDA_VISIBLE_DEVICES=0



## gpu

python /maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/tests/test_gpu.py
nohup python /maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/tests/test_gpu.py > test_gpu.log 2>&1 &

bash
mkdir -p /maindata/data/shared/public/yangchao.zhou/projects/new_tmp

export TMPDIR=/maindata/data/shared/public/yangchao.zhou/projects/new_tmp

## 两个小时后启动

at-V

sudoserviceatdstart

sudoserviceatdstatus

echo "nohup /maindata/data/shared/public/yangchao.zhou/anaconda3/envs/mistral/bin/python /maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/tests/test_gpu_mem.py > test_gpu_mem.log 2>&1 &" | at now + 1 hour
