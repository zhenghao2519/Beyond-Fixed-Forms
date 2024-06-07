# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict

# segment anything
from segment_anything import build_sam, SamPredictor

import cv2
import numpy as np
import matplotlib.pyplot as plt

# other imports
from PIL import Image
import os
import torch
from tqdm import tqdm
import argparse
import pycocotools.mask
import yaml
import supervision as sv
from huggingface_hub import hf_hub_download
from configs import config
from munch import Munch
import requests

device = torch.device('cuda' if torch.cuda.is_available() 
                          else 'mps' if torch.backends.mps.is_available() 
                          else 'cpu')

def download_file(url, filename):
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Open the file in binary write mode and save the content to the file
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully: {filename}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
        

def load_model_hf(repo_id, filename, ckpt_config_filename, device=device):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    args.device = device
    print(f'printing: {args}')
    model = build_model(args)
    
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    # print("cache file",cache_file)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model

def load_grounded_sam():
    # load grounding dino
    ckpt_repo_id = cfg.ckpt_repo_id
    ckpt_filename = cfg.ckpt_filename
    ckpt_config_filename = cfg.ckpt_config_filename
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename, device)
    # load sam
    sam_checkpoint = cfg.sam_checkpoint # wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    if not os.path.exists(sam_checkpoint):
        url = cfg.sam_url
        filename = cfg.sam_checkpoint
        download_file(url, filename)
    sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    return groundingdino_model, sam_predictor

# detect object using grounding DINO
def detect(image_source, image, text_prompt, model, box_threshold = 0.4, text_threshold = 0.4, device = 'cuda'):
  boxes, logits, phrases = predict(
      model=model, 
      image=image, 
      caption=text_prompt,
      box_threshold=box_threshold,
      text_threshold=text_threshold,
      device=device
  )
  annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
  annotated_frame = annotated_frame[...,::-1] # BGR to RGB 
#   print(f"Detected {len(boxes)} boxes")
  return annotated_frame, boxes 

# Segment and draw masks using SAM
def segment(image, sam_model, boxes):
  if len(boxes) == 0:
      print("No boxes detected")
      return None
  sam_model.set_image(image)
  H, W, _ = image.shape
  boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

  transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(device), image.shape[:2]) # boxes in xyxy format
  masks, _, _ = sam_model.predict_torch(                    # masks in [0, 1] range
      point_coords = None,
      point_labels = None,
      boxes = transformed_boxes,
      multimask_output = False,                          # if True, returns masks for each box
      )
  if masks is None:
      print("No masks generated")
  return masks.cpu() if masks is not None else None

def draw_mask(mask, image, random_color=True):  
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:] 
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")     #annotated fram in PIL format
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

def inference_grounded_sam(image_paths, base_prompt,  dino_box_threshold, dino_text_threshold, mask_2d_dir, draw=True):
    """ Inference using Grounding DINO and SAM
    Args:
        image_path (_type_): path of the image
        base_prompt (_type_): text query
    Returns:
        annotated_frame: frame with detected boxes from grounding DINO
        detected_boxes: detected boxes from grounding DINO
        segmented_frame_masks: masks from SAM
    """
    groundingdino_model, sam_predictor = load_grounded_sam()
    print("Loaded models")
    results = []
    num_img_with_boxes = 0
    for i, image_path in enumerate(tqdm(image_paths, desc="Processing images", leave=False)):
        frame_id = image_path.split('/')[-1]                    # get frame id from image path
        image_source, image = load_image(image_path)
        if image_source is None or image is None:               # skip the image if not loaded
            continue
        annotated_frame, detected_boxes = detect(image_source, image, text_prompt=base_prompt, model=groundingdino_model.to(device), box_threshold= dino_box_threshold, text_threshold=dino_text_threshold)
        if detected_boxes is None or len(detected_boxes) == 0:  # skip the image if no boxes detected
            continue
        num_img_with_boxes += 1
        segmented_frame_masks = segment(image_source, sam_predictor, boxes=detected_boxes)
        # annotated_frame_with_mask = draw_mask(segmented_frame_masks[0][0], annotated_frame)
        # segmented_frame_masks_rle = masks_to_rle(segmented_frame_masks)
        
        if draw:
            annotated_frame_with_mask = annotated_frame
            for i in range(segmented_frame_masks.shape[0]):
                annotated_frame_with_mask = draw_mask(segmented_frame_masks[i][0], annotated_frame_with_mask)
                
            if not os.path.exists(os.path.join(mask_2d_dir, "vis_masks")):
                os.makedirs(os.path.join(mask_2d_dir, "vis_masks"))
            Image.fromarray(annotated_frame_with_mask).save(os.path.join(mask_2d_dir, "vis_masks", frame_id+".png"))
            
            
            
        results.append({
            "frame_id": frame_id, 
            "segmented_frame_masks": segmented_frame_masks # (M, 1, W, H)
            # "segmented_frame_masks_rle": segmented_frame_masks_rle
            })   # add keys "annotated_frame", "detected_boxes" if needed
        
    print(f'Number of images with at least one box detected: {num_img_with_boxes}')
    return results

def get_parser():
    parser = argparse.ArgumentParser(description="Configuration Open3DIS")
    parser.add_argument("--config", type=str, required=True, help="Config")
    return parser

def masks_to_rle(masks) -> dict:
    """
    Encode 2D mask to RLE (save memory and fast)
    """
    res = []
    if masks == None:
        return None
    masks = masks.squeeze(1) # remove batch dimension
    for mask in masks:
        if torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()
        assert isinstance(mask, np.ndarray)
        rle = pycocotools.mask.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("utf-8")
        res.append(rle)
    return res

if __name__ == "__main__":
    
    args = get_parser().parse_args()
    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read())) # Can run in command line using "python your_script.py --config config.yaml"
    device = torch.device('cuda' if torch.cuda.is_available() 
                          else 'mps' if torch.backends.mps.is_available() 
                          else 'cpu')
    scene_id = cfg.scene_id
    root_dir = cfg.root_dir
    image_dir = os.path.join(root_dir, scene_id, "color")
    text_prompt = cfg.base_prompt
    mask_2d_dir = cfg.mask_2d_dir

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    image_files.sort(key=lambda x: int(x.split('.')[0]))         # sort numerically, 1.jpg, 2.jpg, 3.jpg ...
    downsampled_image_files = image_files[::10]                  # get one image every 10 frames
    print(f'Number of downsampled_images:{len(downsampled_image_files)}')
    downsampled_images_paths = [os.path.join(image_dir, f) for f in downsampled_image_files]

    dino_box_threshold = cfg.dino_box_threshold
    dino_text_threshold = cfg.dino_text_threshold
    with torch.cuda.amp.autocast():
        grounded_sam_results = inference_grounded_sam(downsampled_images_paths, text_prompt, dino_box_threshold, dino_text_threshold, mask_2d_dir)
        os.makedirs(mask_2d_dir, exist_ok=True)
        mask_2d_path = os.path.join(mask_2d_dir, f"{scene_id}.pth")
        torch.save(grounded_sam_results, mask_2d_path)   # save all segmented frame masks in a file
        torch.cuda.empty_cache()

    # # Proces every scene (during evaluation)
    # with torch.cuda.amp.autocast(enabled=cfg.fp16):
    #     for scene_id in tqdm(scene_ids):
    #         # Tracker
    #         # done = False
    #         # path = scene_id + ".pth"
    #         # with open("tracker_2d.txt", "r") as file:
    #         #     lines = file.readlines()
    #         #     lines = [line.strip() for line in lines]
    #         #     for line in lines:
    #         #         if path in line:
    #         #             done = True
    #         #             break
    #         # if done == True:
    #         #     print("existed " + path)
    #         #     continue
    #         # # Write append each line
    #         # with open("tracker_2d.txt", "a") as file:
    #         #     file.write(path + "\n")

    #         # if os.path.exists(os.path.join(save_dir, f"{scene_id}.pth")): 
    #         #     print(f"Skip {scene_id} as it already exists")
    #         #     continue

    #         print("Process", scene_id)
    #         grounded_data_dict, grounded_features = gen_grounded_mask_and_feat(
    #             scene_id,
    #             clip_adapter,
    #             clip_preprocess,
    #             grounding_dino_model,
    #             sam_predictor,
    #             class_names=class_names,
    #             cfg=cfg,
    #         )

    #         # Save PC features
    #         torch.save({"feat": grounded_features}, os.path.join(save_dir_feat, scene_id + ".pth"))
    #         # Save 2D mask
    #         torch.save(grounded_data_dict, os.path.join(save_dir, scene_id + ".pth"))
              # empty cache
    #         torch.cuda.empty_cache()