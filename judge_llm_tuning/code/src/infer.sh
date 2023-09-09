CUDA_VISIBLE_DEVICE=0 python cli_wo_history.py \
    --model_name_or_path /home/rayjue/SHARE/rayjue/LLM_MODEL/Qwen-7B \
    --template judge \
    --finetuning_type lora \
    --checkpoint_dir /home/rayjue/SHARE/rayjue/trained \
    --quantization_bit 4