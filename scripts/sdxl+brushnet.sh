export PYTHONUSERBASE=intentionally-disabled;
conda activate testbrush;
MASTER_PORT=$((10000 + RANDOM % 10000))
EXP_NAME="sdxl+brushnet2step"

# 2 step
CUDA_VISIBLE_DEVICES=0 python inference_sdxl_turbo_brushnet.py \
    --num_inference_steps 2 \
    --guidance_scale 0.0 \
    --exp_name $EXP_NAME

# 4 step
EXP_NAME="sdxl+brushnet4step"
CUDA_VISIBLE_DEVICES=0 python inference_sdxl_turbo_brushnet.py \
    --num_inference_steps 4 \
    --guidance_scale 0.0 \
    --exp_name $EXP_NAME
