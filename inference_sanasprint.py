import argparse
import json
import time
import torch
import copy
import sys
import numpy as np
from safetensors.torch import load_file

from pipeline import SanaSprintPipelineInpaintingNoBlend
from PIL import Image
from tqdm import tqdm
from utils import *
from diffusers.image_processor import PixArtImageProcessor
from diffusers import SanaTransformer2DModel
from torchvision.utils import save_image
from PIL import Image
import torch

# from pipeline_sana_sprint import SanaSprintPipeline
from transformers import AutoTokenizer, Gemma2Model
from diffusers import (
    AutoencoderDC, 
    FlowMatchEulerDiscreteScheduler,
    SanaTransformer2DModel,
    DPMSolverMultistepScheduler
)
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
    convert_unet_state_dict_to_peft
)
from peft import LoraConfig, set_peft_model_state_dict

import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from sklearn.mixture import GaussianMixture


COMPLEX_HUMAN_INSTRUCTION = [
    "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
]


class SanaTrigFlow(SanaTransformer2DModel):
    def __init__(self, original_model, guidance=False):
        self.__dict__ = original_model.__dict__
        self.hidden_size = self.config.num_attention_heads * self.config.attention_head_dim
        self.guidance = guidance
        if self.guidance:
            hidden_size = self.config.num_attention_heads * self.config.attention_head_dim
            self.logvar_linear = torch.nn.Linear(hidden_size, 1)
            torch.nn.init.xavier_uniform_(self.logvar_linear.weight)
            torch.nn.init.constant_(self.logvar_linear.bias, 0)

    def forward(
        self, hidden_states, encoder_hidden_states, timestep, guidance=None, **kwargs
    ):
        batch_size = hidden_states.shape[0]
        latents = hidden_states
        prompt_embeds = encoder_hidden_states
        t = timestep

        # TrigFlow --> Flow Transformation
        timestep = t.expand(latents.shape[0]).to(prompt_embeds.dtype)
        latents_model_input = latents

        flow_timestep = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))

        flow_timestep_expanded = flow_timestep.view(-1, 1, 1, 1)
        latent_model_input = latents_model_input * torch.sqrt(
            flow_timestep_expanded**2 + (1 - flow_timestep_expanded) ** 2
        )
        latent_model_input = latent_model_input.to(prompt_embeds.dtype)

        # forward in original flow
        model_out = super().forward(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=flow_timestep,
            guidance=guidance,
            **kwargs,
        )[0]

        # Flow --> TrigFlow Transformation
        trigflow_model_out = (
            (1 - 2 * flow_timestep_expanded) * latent_model_input
            + (1 - 2 * flow_timestep_expanded + 2 * flow_timestep_expanded**2) * model_out
        ) / torch.sqrt(flow_timestep_expanded**2 + (1 - flow_timestep_expanded) ** 2)

        if self.guidance and guidance is not None:
            timestep, embedded_timestep = self.time_embed(
                timestep, guidance=guidance, hidden_dtype=hidden_states.dtype
            )
        else:
            timestep, embedded_timestep = self.time_embed(
                timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )

        return trigflow_model_out


@torch.no_grad()
def process_and_encode(
    src_img, pipe,
    height=1024, width=1024,
    device="cuda:0", weight_dtype=torch.float32
):
    image_processor = PixArtImageProcessor(vae_scale_factor=pipe.vae_scale_factor)
    init_image = image_processor.preprocess([src_img], height=height, width=width)
    
    # vae works well for float32
    init_image = init_image.to(device=device, dtype=torch.float32)
    vae = pipe.vae.to(dtype=torch.float32)

    image_latents = vae.encode(init_image).latent
    image_latents = image_latents.to(dtype=weight_dtype) * pipe.vae.config.scaling_factor

    # switch back to bf16
    pipe.vae.to(dtype=weight_dtype)
    return init_image, image_latents


def process_input_mask(mask_path, target_size=1024):
    mask_img = np.array(Image.open(mask_path).resize((target_size, target_size)).convert("RGB"))
    mask_img = np.sum(mask_img, axis=2)
    mask_img = np.where(mask_img!=0, 1, 0)
    
    return mask_img


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
        default=None,
        required=True,
        help="guidance scale",
    )
    parser.add_argument(
        "--generator_seed",
        type=int,
        default=42,
        help="seed",
    )

    ### Model Loading
    parser.add_argument(
        "--exp_name",
        type=str,
        default="lol",
        help="sana multistep scheduler",
    )

    return parser.parse_args()


@torch.no_grad()
def vae_encode(init_image, pipe, weight_dtype=torch.float32):
    vae = pipe.vae.to(dtype=torch.float32)

    image_latents = vae.encode(init_image).latent
    image_latents = image_latents.to(dtype=weight_dtype) * pipe.vae.config.scaling_factor

    # switch back to bf16
    pipe.vae.to(dtype=weight_dtype)

    return image_latents


@torch.no_grad()
def vae_decode(latents, vae):
    latents = latents.to(vae.dtype)
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    return image


if __name__ == "__main__":
    args = parse_args()
    negative_prompt = "blurry, low quality, low resolution, bad anatomy, bad hands, bad feet, poorly drawn, extra limbs, missing limbs, disfigured, deformed, mutated, cloned face, fused fingers, long neck, unrealistic, unnatural, watermark"

    # Define sana inpaint pipeline
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(args.generator_seed)

    base_data_dir = "checkpoint"
    json_brushnet_data_path = "checkpoint/mapping_file.json"

    # read data json
    with open(json_brushnet_data_path, "r") as fp:
        mapping_file = json.load(fp)

    pipe = SanaSprintPipelineInpaintingNoBlend.from_pretrained(
        "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
        torch_dtype=torch.bfloat16, 
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    vae = AutoencoderDC.from_pretrained(
        "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
        subfolder="vae",
        revision=None,
        variant=None,
    )

    fix_timestep = torch.ones((1)) * 0.5
    t = fix_timestep.view(-1, 1, 1, 1).to(device)
    scm_cfg_scale = torch.tensor(
        np.random.choice([4, 4.5, 5], size=1, replace=True),
        device=device,
    )

    speed_list = []
    with torch.no_grad():
        for it, (key, sample) in tqdm(enumerate(mapping_file.items())):
            if it >= 200:
                break

            img_path = osp.join(base_data_dir, sample["image"])
            id_img = sample["image"].split("/")[-1].split(".")[0]
            mask_read = sample["inpainting_mask"]

            edit_p = sample["caption"]
            print("NORMAL: ", edit_p)

            # Read source image for inpainting
            src_img = Image.open(img_path).convert("RGB").resize(
                (1024, 1024), Image.Resampling.LANCZOS
            )

            complex_human_instruction = COMPLEX_HUMAN_INSTRUCTION
            # process mask for blending
            mask = rle2mask(mask_read, (1024, 1024))

            # Inpaint image
            start_time = time.time()
            inpaint_image_stage_1 = pipe(
                prompt=edit_p,
                generator=generator,
                src_image=src_img,
                guidance_scale=args.guidance_scale,
                mask=mask,
                height=1024,
                width=1024,
                num_inference_steps=args.num_inference_steps,
                complex_human_instruction=complex_human_instruction
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

