export PYTHONUSERBASE=intentionally-disabled;
conda activate testsana;
MASTER_PORT=$((10000 + RANDOM % 10000))
EXP_NAME="sdxl+2step"

# 2 step
CUDA_VISIBLE_DEVICES=0 python inference_sdxl_turbo.py \
    --num_inference_steps 2 \
    --guidance_scale 0.0 \
    --exp_name $EXP_NAME

# 4 step
EXP_NAME="sdxl+4step"
CUDA_VISIBLE_DEVICES=0 python inference_sdxl_turbo.py \
    --num_inference_steps 4 \
    --guidance_scale 0.0 \
    --exp_name $EXP_NAME
