import numpy as np
import os
import os.path as osp
import cv2
import torch



def make_dirs(dir_path):
    if not osp.exists(dir_path):
        os.makedirs(dir_path)


def rle2mask(mask_rle, shape): # height, width
    original_shape = (512, 512) # this is the correct original shape, resize should be processed later
    starts, lengths = [np.asarray(x, dtype=int) for x in (mask_rle[0:][::2], mask_rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    binary_mask = np.zeros(original_shape[0] * original_shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        binary_mask[lo:hi] = 1
    
    binary_mask = binary_mask.reshape(
        original_shape
    )
    resized_binary_mask = cv2.resize(
        binary_mask, shape, interpolation=cv2.INTER_AREA
    )

    return resized_binary_mask

def make_dirs(dir_path):
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

def get_dilated_mask(mask, dilation_size=1, erosion_shape=cv2.MORPH_RECT):

    element = cv2.getStructuringElement(
        erosion_shape,
        (2 * dilation_size + 1, 2 * dilation_size + 1),
        (dilation_size, dilation_size),
    )

    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    dilation_mask = cv2.dilate(mask * 255 / 255, element)

    return dilation_mask

def get_rec_mask(mask):
    act_cols, act_rows = np.where(mask>0)

    (x_min, x_max) = (np.min(act_rows).item(), np.max(act_rows).item())
    (y_min, y_max) = (np.min(act_cols).item(), np.max(act_cols).item())

    canvas = np.zeros(mask.shape)
    canvas[y_min:y_max, x_min:x_max] += 1

    # for visualization only
    # canvas_vis = np.stack([canvas]*3, axis=-1)
    # mask_vis = np.stack([mask]*3, axis=-1)

    # cv2.imwrite("./canvas.png", canvas_vis*255)
    # cv2.imwrite("./mask.png", mask_vis*255)

    return canvas

