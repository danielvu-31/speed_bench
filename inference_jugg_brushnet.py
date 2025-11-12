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

from diffusers import UNet2DConditionModel, LCMScheduler, StableDiffusionXLBrushNetPipeline, BrushNetModel, DPMSolverMultistepScheduler, UniPCMultistepScheduler, AutoencoderKL

import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import torch


def concat_and_save_images(image_list, output_path):
    """
    Concatenates a list of PIL Images and/or torch Tensors horizontally and saves the result.

    Args:
        image_list (list): A list containing a mix of PIL.Image.Image and torch.Tensor objects.
                           All images and tensors are expected to have the same height.
        output_path (str): The path to save the concatenated image.
    """
    tensor_list = []
    
    # Define a transform to convert PIL images to tensors
    pil_to_tensor_transform = transforms.ToTensor()

    for img in image_list:
        if isinstance(img, Image.Image):
            # Convert PIL Image to tensor
            tensor_list.append(pil_to_tensor_transform(img.convert("RGB")).squeeze())
        elif isinstance(img, torch.Tensor):
            # Add tensor to the list
            tensor_list.append(img.squeeze().detach().cpu())
        else:
            raise TypeError(f"Unsupported type in the list: {type(img)}")

    
    # Ensure all tensors have the same height and number of channels
    if len(tensor_list) > 1:
        first_tensor = tensor_list[0]
        for i in range(1, len(tensor_list)):
            if tensor_list[i].shape[1] != first_tensor.shape[1] or tensor_list[i].shape[0] != first_tensor.shape[0]:
                raise ValueError("All images and tensors must have the same height and number of channels.")

    # Concatenate tensors horizontally (along the width dimension, which is axis 2)
    concatenated_tensor = torch.cat(tensor_list, dim=2)

    # Save the concatenated tensor as an image
    save_image(concatenated_tensor, output_path)
    print(f"Concatenated image saved to {output_path}")


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

    ###### BrushNet ######
    parser.add_argument(
        "--brushnet_mode",
        type=str,
        default="random",
        choices=["segment", "random"],
    )

    ###### Inversion ######
    parser.add_argument(
        "--exp_name",
        type=str,
        default="lol",
    )

    return parser.parse_args()

    return parser.parse_args()


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
    
    weight_dtype = torch.bfloat16

    base_data_dir = "checkpoint"
    json_brushnet_data_path = "checkpoint/mapping_file.json"

    sdxl_path = "misri/juggernautXL_juggernautX"
    
    brushnet_path = "checkpoint/brushnet/random_mask_brushnet_ckpt_sdxl_v0"
    brushnet = BrushNetModel.from_pretrained(
        brushnet_path,
        torch_dtype=weight_dtype
    ).to(device)

    pipe = StableDiffusionXLBrushNetPipeline.from_pretrained(
        sdxl_path,
        brushnet=brushnet,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to("cuda")

    # read data json
    with open(json_brushnet_data_path, "r") as fp:
        mapping_file = json.load(fp)

    speed_list = []
    brushnet_conditioning_scale = 1.0
    with torch.no_grad():
        for it, (key, sample) in tqdm(enumerate(mapping_file.items())):
            if it >= 200:
                break

            img_path = osp.join(base_data_dir, sample["image"])
            id_img = sample["image"].split("/")[-1].split(".")[0]
            mask_read = sample["inpainting_mask"]

            edit_p = sample["caption"]
            print("Normal: ", edit_p)
            
            src_img = cv2.imread(img_path)[:, :, ::-1]
            src_img = cv2.resize(src_img, (1024, 1024))

            # process mask for blending
            mask = rle2mask(mask_read, (1024, 1024))
            mask = mask[:, :, np.newaxis]

            init_image = src_img * (1 - mask)
            init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
            mask_image = Image.fromarray(mask.astype(np.uint8).repeat(3, -1) * 255).convert("RGB")

            start_time = time.time()
            image = pipe(
                image=init_image,
                mask=mask_image,
                prompt=edit_p,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_prompt=negative_prompt,
                negative_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                height=1024,
                width=1024,
                brushnet_conditioning_scale=brushnet_conditioning_scale,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator
            ).images[0]

            run_time = time.time() - start_time
            print(f"Finish inpainting within {run_time}")
            if it >= 100 and it < 200:
                speed_list.append(run_time)
    
    mean_runtime = np.mean(speed_list)
    print(f"AVERAGE: ", np.mean(speed_list))
    with open(f"{args.exp_name}.txt", "w") as f:
        f.write(f"AVERAGE: {mean_runtime:.16f}\n")
        f.close()

