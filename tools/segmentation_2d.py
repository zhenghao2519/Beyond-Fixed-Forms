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
import os
import torch
import supervision as sv
from huggingface_hub import hf_hub_download


# TODO: TO BE ORGANIZED TO A SEPERATE FILE FOR CONFIGURING DIFFERENT DATASETS AND SCENES, ESP. FOR EVALUATION
class Config:
    def __init__(self):
        self.dataset = 'Scannet200'
        self.scene_id = 'scene0435_00' # scannet scene id
        self.root_path = '/home/jie_zhenghao/Open3DIS/data/Scannet200/Scannet200_2D_5interval/val'
        self.scene_dir = os.path.join(cfg.root_path, self.scene_id)
        self.image_dir = '/home/jie_zhenghao/Open3DIS/data/Scannet200/Scannet200_2D_5interval/val/scene0435_00/color/'
        
        self.num_workers = 4
        self.batch_size = 10
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        # grounding dino
        self.ckpt_repo_id = "ShilongLiu/GroundingDINO"
        self.ckpt_filename = "groundingdino_swinb_cogcoor.pth"
        self.ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        # sam
        self.sam_checkpoint = 'sam_vit_h_4b8939.pth'
        self.base_prompt = "clothes" #, Pillow, Chair, Sofa, Bed, Desk, Monitor, Television, Book"

def load_model_hf(repo_id, filename, ckpt_config_filename, device=device):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    args.device = device
    print(f'printing: {args}')
    model = build_model(args)
    
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model

def load_grounded_sam():
    # load grounding dino
    ckpt_repo_id = cfg.ckpt_repo_id
    ckpt_filenmae = cfg.ckpt_config_filename
    ckpt_config_filename = cfg.ckpt_config_filename
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)
    # load sam
    sam_checkpoint = cfg.sam_checkpoint
    sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    return groundingdino_model, sam_predictor

# detect object using grounding DINO
def detect(image_source, image, text_prompt, model, box_threshold = 0.3, text_threshold = 0.3, device = 'cuda'):
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

# Process images
def inference_grounded_sam(image_paths, base_prompt):
    groundingdino_model, sam_predictor = load_grounded_sam()
    print("Loaded models")
    results = []
    num_img_with_boxes = 0
    for i, image_path in enumerate(tqdm(image_paths, desc="Processing images", leave=False)):
        image_source, image = load_image(image_path)
        if image_source is None or image is None:
            # tqdm.write(f'Failed to load image: {image_path}')
            continue
        annotated_frame, detected_boxes = detect(image_source, image, text_prompt=base_prompt, model=groundingdino_model.to(device))
        if detected_boxes is None or len(detected_boxes) == 0:
            # tqdm.write(f"No boxes detected for image{image_path}")
            continue
        num_img_with_boxes += 1
        segmented_frame_masks = segment(image_source, sam_predictor, boxes=detected_boxes)
        # annotated_frame_with_mask = draw_mask(segmented_frame_masks[0][0], annotated_frame)
        # results.append((image_path, annotated_frame_with_mask))
    print(f'Number of images with at least one box detected: {num_img_with_boxes}')
    return annotated_frame, segmented_frame_masks
  
# def draw_mask(mask, image, random_color=True):  
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
#     else:
#         color = np.array([30/255, 144/255, 255/255, 0.6])
#     h, w = mask.shape[-2:] 
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
#     annotated_frame_pil = Image.fromarray(image).convert("RGBA")     #annotated fram in PIL format
#     mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

#     return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

# Data loading function to create batches of image paths
# def dataloader(image_dir, batch_size=10):
#     image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
#     image_files.sort(key=lambda x: int(x.split('.')[0]))  # Sort numerically
#     downsampled_image_files = image_files[::10]  # Downsample by taking one every 10 frames
#     print(f'Number of downsampled images: {len(downsampled_image_files)}')
#     for i in range(0, len(downsampled_image_files), batch_size):
#         yield [os.path.join(image_dir, f) for f in downsampled_image_files[i:i + batch_size]]


if __name__ == "__main__":

    cfg = Config()
    device = cfg.device
    scene_id = cfg.scene_id
    image_dir = cfg.image_dir
    batch_size = cfg.batch_size
    base_prompt = cfg.base_prompt
    save_dir = os.path.join("../exp",  'segmented_masks')

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    image_files.sort(key=lambda x: int(x.split('.')[0]))         # sort numerically
    downsampled_image_files = image_files[::10]                  # get one image every 10 frames
    print(f'Number of downsampled_images:{len(downsampled_image_files)}')
    downsampled_images_paths = [os.path.join(image_dir, f) for f in downsampled_image_files]

   
    with torch.cuda.amp.autocast(enable = True):
        _, segmented_frame_masks = inference_grounded_sam(downsampled_images_paths, base_prompt)
        os.makedirs(save_dir, exist_ok=True)
        path = scene_id + ".pth"
        torch.save(segmented_frame_masks, os.path.join(save_dir, path))   # save all segmented frame masks in a file
        torch.cuda.empty_cache()

    # # Proces every scene
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

    #         torch.cuda.empty_cache()