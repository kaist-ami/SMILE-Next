# # LLaMA3-8B
CUDA_VISIBLE_DEVICES=0,2 llamafactory-cli train llamafactory_configs/llama3_selfinst_moelora_sft_ds3.yaml

# # Qwen2.5-7B
CUDA_VISIBLE_DEVICES=2,3 FORCE_TORCHRUN=1 llamafactory-cli train llamafactory_configs/qwen25_selfinst_moelora_sft_ds3.yaml