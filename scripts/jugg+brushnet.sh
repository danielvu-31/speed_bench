export PYTHONUSERBASE=intentionally-disabled;
conda activate testbrush;
MASTER_PORT=$((10000 + RANDOM % 10000))
EXP_NAME="jugg-brushnet30step+cfg8.0"

# 2 step
CUDA_VISIBLE_DEVICES=0 python inference_jugg_brushnet.py \
    --num_inference_steps 30 \
    --guidance_scale 8.0 \
    --exp_name $EXP_NAME