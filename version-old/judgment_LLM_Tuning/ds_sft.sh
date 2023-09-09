accelerate launch src/train_sft.py \
    --model_name_or_path /hy-tmp/LLM/lama-7b \
    --dataset merge.json,judgement.json,COT_chinese.json  \
    --do_train \
    --finetuning_type full \
    --output_dir /hy-tmp/saved/new_full/judgement \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-3 \
    --num_train_epochs 15.0 \
    --plot_loss \
    --fp16