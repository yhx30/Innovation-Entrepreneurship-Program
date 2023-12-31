accelerate launch src/train_sft.py \
    --model_name_or_path /home/rayjue/SHARE/rayjue/LLM_MODEL/chatglm2-6b-32k \
    --dataset judgement  \
    --do_train \
    --finetuning_type full \
    --output_dir /home/rayjue/SHARE/rayjue/trained \
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