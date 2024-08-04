num_frames=16
test_ratio=1
model_dir=MODELS/pllava-7b
weight_dir=MODELS/pllava-7b
SAVE_DIR=test_results/test_pllava_7b
lora_alpha=4
conv_mode=eval_recaption

python -m pllava_caption \
    --pretrained_model_name_or_path ${model_dir} \
    --save_path ${SAVE_DIR}/recaption \
    --num_frames ${num_frames} \
    --use_lora \
    --lora_alpha ${lora_alpha} \
    --weight_dir ${weight_dir} \
    --test_ratio ${test_ratio}