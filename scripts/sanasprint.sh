export PYTHONUSERBASE=intentionally-disabled;
conda activate testsana;
MASTER_PORT=$((10000 + RANDOM % 10000))
EXP_NAME="sanasprint-2steps"

# 2 step
CUDA_VISIBLE_DEVICES=0 python inference_sanasprint.py \
    --num_inference_steps 2 \
    --guidance_scale 4.5 \
    --exp_name $EXP_NAME

# 4 step
EXP_NAME="sanasprint-4steps"
CUDA_VISIBLE_DEVICES=0 python inference_sanasprint.py \
    --num_inference_steps 4 \
    --guidance_scale 4.5 \
    --exp_name $EXP_NAME
