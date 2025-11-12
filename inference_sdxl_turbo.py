import argparse
import json
import time
import torch
import copy
import sys
from safetensors.torch import load_file

from PIL import Image
from tqdm import tqdm
from utils import *
from torchvision.utils import save_image
from diffusers.image_processor import VaeImageProcessor

from transformers import AutoTokenizer
from diffusers import UNet2DConditionModel, LCMScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler, AutoencoderKL
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.utils import (
    check_min_version,
    load_image,
    is_wandb_available,
    convert_unet_state_dict_to_peft
)

import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import torch

from pipeline_sdxl import StableDiffusionXLPipeline


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Infer samples from sana sprint."
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        required=True,
        help="Num inference steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=0.0,
        required=True,
        help="guidance scale",
    )
    parser.add_argument(
        "--generator_seed",
        type=int,
        default=42,
        help="seed",
    )

    ###### Inversion ######
    parser.add_argument(
        "--exp_name",
        type=str,
        default="lol",
    )

    return parser.parse_args()


@torch.no_grad()
def process_and_encode(
    src_img, pipe, height=1024, width=1024,
    device="cuda:0", weight_dtype=torch.bfloat16
):
    image_processor = VaeImageProcessor(vae_scale_factor=pipe.vae_scale_factor)
    init_image = image_processor.preprocess([src_img], height=height, width=width)
    
    # vae works well for float32
    init_image = init_image.to(device=device, dtype=torch.float32)
    vae = pipe.vae.to(dtype=torch.float32)

    image_latents = vae.encode(init_image).latent_dist
    image_latents = image_latents.sample().to(dtype=weight_dtype) * pipe.vae.config.scaling_factor

    # switch back to bf16
    pipe.vae.to(dtype=weight_dtype)
    return init_image, image_latents


def get_add_time_ids(
    original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
):
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    return add_time_ids


@torch.no_grad()
def vae_decode(latents, vae):
    latents = latents.to(vae.dtype)
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    return image


@torch.no_grad()
def vae_encode(images, vae, weight_dtype=torch.bfloat16):
    vae = vae.to(torch.float32)
    images = images.to(torch.float32)

    latents = vae.encode(images).latent_dist.sample() *  vae.config.scaling_factor
    vae = vae.to(weight_dtype)

    return latents


if __name__ == "__main__":
    args = parse_args()
    negative_prompt = "blurry, low quality, low resolution, bad anatomy, bad hands, bad feet, poorly drawn, extra limbs, missing limbs, disfigured, deformed, mutated, cloned face, fused fingers, long neck, unrealistic, unnatural, watermark"
    
    weight_dtype = torch.bfloat16
    
    # Define sana inpaint pipeline
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(args.generator_seed)

    base_data_dir = "checkpoint"
    json_brushnet_data_path = "checkpoint/mapping_file.json"

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", 
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to("cuda")

    # read data json
    with open(json_brushnet_data_path, "r") as fp:
        mapping_file = json.load(fp)
    
    speed_list = []
    with torch.no_grad():
        for it, (key, sample) in tqdm(enumerate(mapping_file.items())):
            if it >= 200:
                break

            img_path = osp.join(base_data_dir, sample["image"])
            id_img = sample["image"].split("/")[-1].split(".")[0]
            mask_read = sample["inpainting_mask"]

            edit_p = sample["caption"]
            print("Normal: ", edit_p)
            
            src_img = Image.open(img_path).convert("RGB").resize(
                (1024, 1024), Image.Resampling.LANCZOS
            )
            mask = rle2mask(mask_read, (1024, 1024))
            mask_image = Image.fromarray((mask * 255).astype(np.uint8))

            #### Start ####
            start_time = time.time()
            image = pipe(
                prompt=edit_p,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_prompt=None,
                negative_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                src_image=src_img,
                init_image=None,
                mask_image=mask,
                height=1024,
                width=1024,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator
            ).images[0]
            run_time = time.time() - start_time

            if it >= 100 and it < 200:
                speed_list.append(run_time)
            print(f"Finish inpainting within {run_time}")
    
    mean_runtime = np.mean(speed_list)
    print(f"AVERAGE: ", np.mean(speed_list))
    with open(f"{args.exp_name}.txt", "w") as f:
        f.write(f"AVERAGE: {mean_runtime:.16f}\n")
        f.close()


#### python speed/inference_kien_sdxl_turbo_brushnet.py --num_inference_steps 2 --guidance_scale 0.0 --checkpoint_path "/lustre/scratch/client/movian/research/users/ducvh5/sana_lora_inpainting/sdxl_exps/kien_reg-mean+var0.5_fullparams1e-5_maskimage+gaussianblend+losspost_1step_randommask_clsgan+lr1e-6+weight0.5+1critic/checkpoint-3000" --fill_mask_type "gaussian" --input_mask_type "image" --use_ema