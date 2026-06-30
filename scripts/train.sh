# # Qwen2.5-Omni-7B
# CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train llamafactory_configs/qwen25omni_lora_sft.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train llamafactory_configs/qwen25omni_lora_sft_gpu11.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli chat llamafactory_configs/inference/qwen25omni_lora_sft_gpu11.yaml

# # LLaMA3-8B
# CUDA_VISIBLE_DEVICES=0,2 llamafactory-cli train llamafactory_configs/llama3_selfinst_lora_sft_ds3.yaml

# # Qwen2.5-VL-7B
# CUDA_VISIBLE_DEVICES=3,5 llamafactory-cli train llamafactory_configs/qwen25vl_lora_sft.yaml

# # Qwen2.5-7B
CUDA_VISIBLE_DEVICES=2,3 FORCE_TORCHRUN=1 llamafactory-cli train llamafactory_configs/qwen25_selfinst_lora_sft_ds3.yaml

# MiniCPM-2.6-o (v, and any other else)
# CUDA_VISIBLE_DEVICES=4 llamafactory-cli train llamafactory_configs/minicpmo_26_lora_sft_ds3.yaml
# CUDA_VISIBLE_DEVICES=4 llamafactory-cli train llamafactory_configs/minicpmo_26_full_sft_ds3.yaml

# Qwen2.5-7B (moelora)
# CUDA_VISIBLE_DEVICES=2,3,5,6 FORCE_TORCHRUN=1 llamafactory-cli train /home/mok/module/ijcv25/models/llamafactory_configs/qwen25_moelora_sft_ds3.yaml

# LLaMA3-8B (moelora)
# CUDA_VISIBLE_DEVICES=2,3,5,6 llamafactory-cli train llamafactory_configs/qwen25_selfinst_moelora_sft_ds3.yaml