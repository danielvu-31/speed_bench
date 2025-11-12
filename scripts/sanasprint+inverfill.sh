export PYTHONUSERBASE=intentionally-disabled;
conda activate testsana;
MASTER_PORT=$((10000 + RANDOM % 10000))
EXP_NAME="sanasprint+inverfill-2steps"

# 2 step
CUDA_VISIBLE_DEVICES=0 python inference_kien_sanasprint.py \
    --num_inference_steps 2 \
    --guidance_scale 4.5 \
    --fill_mask_type "gaussian" \
    --input_mask_type "image" \
    --exp_name $EXP_NAME

# 4 step
EXP_NAME="sanasprint+inverfill-4steps"
CUDA_VISIBLE_DEVICES=0 python inference_kien_sanasprint.py \
    --num_inference_steps 4 \
    --guidance_scale 4.5 \
    --fill_mask_type "gaussian" \
    --input_mask_type "image" \
    --exp_name $EXP_NAME
