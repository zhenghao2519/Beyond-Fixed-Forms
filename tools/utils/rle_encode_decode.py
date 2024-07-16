import numpy as np
import torch
import os
import copy

import sys
sys.path.append("./")
from evaluation.dataset.scannet200 import INSTANCE_CAT_SCANNET_200

def rle_encode_batch(masks):
    """Encode RLE (Run-length-encode) from 1D binary mask.

    Args:
        mask (np.ndarray): 1D binary mask
    Returns:
        rle (dict): encoded RLE
    """
    n_inst, length = masks.shape[:2]
    zeros_tensor = torch.zeros((n_inst, 1), dtype=torch.bool, device=masks.device)
    masks = torch.cat([zeros_tensor, masks, zeros_tensor], dim=1)

    rles = []
    for i in range(n_inst):
        mask = masks[i]
        runs = torch.nonzero(mask[1:] != mask[:-1]).view(-1) + 1

        runs[1::2] -= runs[::2]

        counts = runs.cpu().numpy()
        rle = dict(length=length, counts=counts)
        rles.append(rle)
    return rles


def rle_decode_batch(rles):
    """Decode rle to get binary mask.

    Args:
        rle (dict): rle of encoded mask
    Returns:
        mask (torch.Tensor): decoded mask
    """
    n_inst = len(rles)
    masks = []
    for i in range(n_inst):
        rle = rles[i]
        # print(rle)
        s = rle["counts"]
        length = rle["length"]

        starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + nums
        mask = np.zeros(length, dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            mask[lo:hi] = 1
        masks.append(mask)
        
    # covert to torch
    masks = torch.from_numpy(np.stack(masks))
    return masks

def encode_2d_masks(masks_2d):
    """Encode 2D masks to RLE encoding.
    
    Args:
        path (str): path to 2D masks
        
    Returns:
        masks_2d (list): list of RLE encoded masks
    """
    
    ## covert all masks in mask_2d to rle encoding
    for i in range(len(masks_2d)):
        masks = masks_2d[i]['segmented_frame_masks']
        masks = masks.view(masks.shape[0], -1)
        rles = rle_encode_batch(masks)
        masks_2d[i]['segmented_frame_masks'] = rles
        
    return masks_2d

def decode_2d_masks(masks_2d, image_shape=(968, 1296)):
    """Decode RLE encoding to 2D masks.
    
    Args:
        masks_2d (list): list of RLE encoded masks
        
    Returns:
        masks_2d (list): list of 2D masks
    """
    
    ## covert all masks in mask_2d to rle encoding
    for i in range(len(masks_2d)):
        masks = masks_2d[i]['segmented_frame_masks']
        masks = rle_decode_batch(masks)
        masks = masks.view(masks.shape[0], 1, *image_shape)
        masks_2d[i]['segmented_frame_masks'] = masks
        
    return masks_2d

if __name__ == "__main__":
    # encode 2D masks to rle
    root_dir = "/home/jie_zhenghao/Beyond-Fixed-Forms/output/mask_2d"
    
    classes  = sorted(os.listdir(root_dir))
    # cls = "tv stand"
    for cls in classes:
        
        if cls not in INSTANCE_CAT_SCANNET_200:
            continue
        
        print(f"Processing {cls}...")
        path = os.path.join(root_dir, cls)
        
        output_path = os.path.join(root_dir, "rle", cls)
        os.makedirs(output_path, exist_ok=True)
        
        for file in os.listdir(path):
            masks_2d = torch.load(os.path.join(path, file))
            masks_2d_rle = encode_2d_masks(masks_2d)
            

            
            # # verify the encoding
            # masks_2d_decoded = decode_2d_masks(copy.deepcopy(masks_2d_rle))
            # for i in range(len(masks_2d)):
            #     assert torch.all(masks_2d[i]['segmented_frame_masks'] == masks_2d_decoded[i]['segmented_frame_masks'])
            # print(f"Encoding for {file} is correct.")
            
            torch.save(masks_2d_rle, os.path.join(output_path, file))
            