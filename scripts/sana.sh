export PYTHONUSERBASE=intentionally-disabled;
conda activate testsana;
MASTER_PORT=$((10000 + RANDOM % 10000))
EXP_NAME="sana-20steps"

# 2 step
CUDA_VISIBLE_DEVICES=0 python inference_sana.py \
    --num_inference_steps 20 \
    --guidance_scale 4.5 \
    --exp_name $EXP_NAME
