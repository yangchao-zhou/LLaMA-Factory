export CUDA_LAUNCH_BLOCKING=1

pip install -e .[torch,metrics]

conda activate OpenRLHF

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft_ds2_mistral.yaml

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml


TRANSFORMERS_VERBOSITY=debug FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml
