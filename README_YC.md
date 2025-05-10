## todo

尝试冻结一部分层
可以试试lora 的效果

合并后的数据数量：54579

基于长COT的角色扮演(生成针对回答的推理过程，然后一起训练)
把有comment的打分数据参与训练
试试之前指令模型开始训练的效果

生成聊天数据的数据
如何让history不参与训练
请用长思维链的方式回答。回答之前先思考一下，长思维链的内容在<think>和</think>之间。思维链之外的内容才是真的回答

请用长思维链的方式回答。回答之前先思考一下：
1. 用户的意图，
2. 作为NPC，结合自己特性当下的场景，你应该怎么高情商的回复
长思维链的内容在<think>和</think>之间。
思维链之外的内容才是真的回答。

请用思维链的方式回答每一轮对话。回答之前先思考一下：
1. 用户的意图，
2. 作为NPC，结合自己性格特点，思考你应该怎么回复
思维链的内容在<think>和</think>之间。思维链的部分不要超过100个单词。
思维链之外的内容才是真的回答。

请用思维链的方式回答**每一轮对话**。回答之前先思考一下：
1. 用户的意图，
2. 作为NPC，结合自己性格特点，NPC intro,NPC greeting，(用户已经读过NPC intro,NPC greeting )思考你应该怎么回复，

尽量要勾起用户的参与，让用户开口说更多的话，让他们感兴趣。你们的对话才可以一直聊下去。但是不要有什么离谱的情节和生硬的情节转折
思维链的内容在<think>和</think>之间。思维链的部分不要超过100个单词。
思维链之外的内容才是真的回答。


可以虚构NPC自己的背景，但是不能虚构用户的信息
对话应该活灵活现，非常像一个真实的人
intro  和 greeting放在最后


  "role_play_combined_data_202500304_3": {
    "file_name": "/maindata/data/shared/public/yangchao.zhou/projects/mistral_pro/data/need_merge_data/250304_long_COT_MultiTurn/combined_data_20250304_过采样_3_greeting.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "system": "system"
    }

## todo
plan execution Reflection direct_answer

## 训练

### 火山分布式SFT
pkill -f "llamafactory"
watch -n 1 gpustat

pkill -f "llamafactory"
pkill -f "test_gpu_mem"

export  PYTHONPATH=`pwd`
export  PYTHONPATH=/maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory
export  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
export TMPDIR=/maindata/data/shared/public/yangchao.zhou/projects/tmp
export NCCL_SOCKET_IFNAME=eth1

cd /maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/

conda activate nemo

rm -rf train-$MLP_ROLE_INDEX.log

nohup bash -c 'FORCE_TORCHRUN=1 NNODES=$MLP_WORKER_NUM NODE_RANK=$MLP_ROLE_INDEX MASTER_ADDR=$MLP_WORKER_0_HOST MASTER_PORT=$MLP_WORKER_0_PORT  /root/miniconda3/envs/nemo/bin/llamafactory-cli train /maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/examples/train_full/mistral_full_sft_ds.yaml' > train-$MLP_ROLE_INDEX.log 2>&1 &

tail -f train-$MLP_ROLE_INDEX.log

llama3_full_sft_ds
mistral_full_sft_ds

### 单节点PT

```bash
nohup llamafactory-cli train examples/train_full/mistral_full_sft_ds.yaml > train_mistral_full_sft_ds_output.log 2>&1 &
```

### 单节点SFT
```bash
pkill -f "llamafactory"
sudo -s
conda activate nemo

cd /maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install deepspeed==0.14.5
pip install flash-attn==2.6.2
pip install lmdeploy
export  PYTHONPATH=`pwd`

export TMPDIR=/maindata/data/shared/public/yangchao.zhou/projects/tmp

conda activate nemo
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
llamafactory-cli train examples/train_full/mistral_full_sft_ds.yaml

nohup llamafactory-cli train examples/train_full/mistral-24B-ins-sft-AIME_gpqa-diamond_HLE_usamo.yaml > train_output-mistral-24B-ins-sft-AIME_gpqa-diamond_HLE_usamo.log 2>&1 &

nohup llamafactory-cli train examples/train_full/mistral-24B-ins-sft-AIME_gpqa-diamond_HLE_usamo_SWE-bench_Verified.yaml > train_output-mistral-24B-ins-sft-AIME_gpqa-diamond_HLE_usamo_SWE-bench_Verified.log 2>&1 &

nohup llamafactory-cli train examples/train_full/qwq_full_sft_ds.yaml > train_output_qwq-1.log 2>&1 &

sudo chown -R ran.xiao /maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory
sudo chmod -R 777 /maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/

```

### 在多机上进行指令监督微调

```bash
pkill -f "llamafactory"
pkill -f "vllm"
watch -n 1 gpustat
三个节点的公网ip
10.1.16.59
10.1.16.66
10.1.16.77
10.1.16.57

检测是否端口被占
netstat -tulnp | grep 29500

export  PYTHONPATH=`pwd`
export  PYTHONPATH=/maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory
export  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
export TMPDIR=/maindata/data/shared/public/yangchao.zhou/projects/tmp
export NCCL_SOCKET_IFNAME=eth1
 # 根据 ifconfig 结果选择正确的网卡，如 eth1

# FORCE_TORCHRUN=1 NNODES=4 NODE_RANK=0 MASTER_ADDR=10.1.16.59 MASTER_PORT=29500 llamafactory-cli train examples/train_full/qwen_full_sft_ds.yaml
# FORCE_TORCHRUN=1 NNODES=4 NODE_RANK=1 MASTER_ADDR=10.1.16.59 MASTER_PORT=29500 llamafactory-cli train examples/train_full/qwen_full_sft_ds.yaml
# FORCE_TORCHRUN=1 NNODES=4 NODE_RANK=2 MASTER_ADDR=10.1.16.59 MASTER_PORT=29500 llamafactory-cli train examples/train_full/qwen_full_sft_ds.yaml
# FORCE_TORCHRUN=1 NNODES=4 NODE_RANK=3 MASTER_ADDR=10.1.16.59 MASTER_PORT=29500 llamafactory-cli train examples/train_full/qwen_full_sft_ds.yaml

nohup bash -c 'FORCE_TORCHRUN=1 NNODES=3 NODE_RANK=0 MASTER_ADDR=10.1.16.59 MASTER_PORT=29500 llamafactory-cli train examples/train_full/qwen_full_sft_ds.yaml' > train-0.log 2>&1 &
nohup bash -c 'FORCE_TORCHRUN=1 NNODES=3 NODE_RANK=1 MASTER_ADDR=10.1.16.59 MASTER_PORT=29500 llamafactory-cli train examples/train_full/qwen_full_sft_ds.yaml' > train-1.log 2>&1 &
nohup bash -c 'FORCE_TORCHRUN=1 NNODES=3 NODE_RANK=2 MASTER_ADDR=10.1.16.59 MASTER_PORT=29500 llamafactory-cli train examples/train_full/qwen_full_sft_ds.yaml' > train-2.log 2>&1 &


bash -c 'FORCE_TORCHRUN=1 NNODES=$MLP_WORKER_NUM NODE_RANK=$MLP_ROLE_INDEX MASTER_ADDR=$MLP_WORKER_0_HOST MASTER_PORT=$MLP_WORKER_0_PORT  /root/miniconda3/envs/nemo/bin/llamafactory-cli train /maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/examples/train_full/llama3_full_sft_ds.yaml' 

```

### 单节点强化学习
```bash
nohup bash -c 'llamafactory-cli train examples/train_full/mistral_full_rl_ds.yaml' > mistral_full_rl_ds.log 2>&1 &
```

### 在多机上进行强化学习

```bash
pkill -f "llamafactory"
pkill -f "vllm"
watch -n 1 gpustat

export  PYTHONPATH=`pwd`
export  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
export TMPDIR=/maindata/data/shared/public/yangchao.zhou/projects/tmp
export NCCL_SOCKET_IFNAME=eth1

nohup bash -c 'FORCE_TORCHRUN=1 NNODES=4 NODE_RANK=0 MASTER_ADDR=10.1.16.59 MASTER_PORT=29500 llamafactory-cli train examples/train_full/llama3_full_rl_ds.yaml' > train-0.log 2>&1 &
nohup bash -c 'FORCE_TORCHRUN=1 NNODES=4 NODE_RANK=1 MASTER_ADDR=10.1.16.59 MASTER_PORT=29500 llamafactory-cli train examples/train_full/llama3_full_rl_ds.yaml' > train-1.log 2>&1 &
nohup bash -c 'FORCE_TORCHRUN=1 NNODES=4 NODE_RANK=2 MASTER_ADDR=10.1.16.59 MASTER_PORT=29500 llamafactory-cli train examples/train_full/llama3_full_rl_ds.yaml' > train-2.log 2>&1 &
nohup bash -c 'FORCE_TORCHRUN=1 NNODES=4 NODE_RANK=3 MASTER_ADDR=10.1.16.59 MASTER_PORT=29500 llamafactory-cli train examples/train_full/llama3_full_rl_ds.yaml' > train-3.log 2>&1 &

```

## vllm 部署

export  PYTHONPATH=`pwd`
export  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
export TMPDIR=/maindata/data/shared/public/yangchao.zhou/projects/tmp
export NCCL_SOCKET_IFNAME=eth1
pkill -f "vllm"
nohup vllm serve /maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/saves/full/Llama-33-70b-ins-sft_AIME_gpqa-diamond_HLE_usamo_20250406/checkpoint-450\
    --task generate \
    --tensor-parallel-size 8 \
    --port 3280 \
    --served-model-name let_it_out \
    --gpu-memory-utilization 0.9 \
    > vllm_logs/vllm-Llama-33-70b-ins-sft_AIME_gpqa-diamond_HLE_usamo_20250406.log 2>&1 &

pkill -f "llamafactory"
pkill -f "vllm"
pkill -f 'from multiprocessing.spawn import spawn_main'

watch -n 1 gpustat


nohup vllm serve /maindata/data/shared/public/common_models/Llama-3.3-70B-Instruct \
    --task generate \
    --tensor-parallel-size 8 \
    --port 3280 \
    --served-model-name let_it_out \
    --gpu-memory-utilization 0.45 \
    --host 0.0.0.0 \
    > vllm_logs/Llama-3.3-70B-Instruct.log 2>&1 &

nohup vllm serve /maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/saves/full/Llama-33-70b-ins-sft_LCB_Math_20250423-5e-6/checkpoint-610 \
    --task generate \
    --tensor-parallel-size 8 \
    --port 3280 \
    --served-model-name let_it_out \
    --gpu-memory-utilization 0.45 \
    > vllm_logs/vllm-Llama-33-70b-ins-sft_LCB_Math_20250423-5e-6.log 2>&1 &

nohup vllm serve /maindata/data/shared/public/yangchao.zhou/projects/LLaMA-Factory/saves/full_sft/mistral-24B-ins-sft_aider_20250510-5e-6-wocao \
    --task generate \
    --tensor-parallel-size 8 \
    --port 3280 \
    --served-model-name let_it_out \
    --gpu-memory-utilization 0.45 \
    > vllm_logs/vllm-mistral-24B-ins-sft_aider_20250510-5e-6.log 2>&1 &



    
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
```