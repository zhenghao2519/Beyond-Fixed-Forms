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
import torch
import supervision as sv
from huggingface_hub import hf_hub_download

device = torch.device('cuda' if torch.cuda.is_available() 
                          else 'mps' if torch.backends.mps.is_available() 
                          else 'cpu')

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
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)

    # load sam
    sam_checkpoint = 'sam_vit_h_4b8939.pth' #!! change this to the path of the checkpoint
    sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    return groundingdino_model, sam_predictor

# detect object using grounding DINO
def detect(image, text_prompt, model, box_threshold = 0.35, text_threshold = 0.35):
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
    return annotated_frame, boxes 

def segment(image, sam_model, boxes):
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
  return masks.cpu()
  

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

def inference_grounded_sam():
    groundingdino_model, sam_predictor = load_grounded_sam()
    print("Loaded models")

    # Load image
    local_image_path = '../data/Scannet200/Scannet200_2D_5interval/val/scene0435_00/color/738.jpg' #"assets/demo9.jpg" 738
    image_source, image = load_image(local_image_path)

    # Predict grounding
    base_prompt = "Clothes, Pillow, Chair, Sofa, Bed, Desk, Monitor, Television, Book"
    annotated_frame, detected_boxes = detect(image, text_prompt=base_prompt, model=groundingdino_model.to(device))

    # Predict segmentation
    segmented_frame_masks = segment(image_source, sam_predictor, boxes=detected_boxes)

    return annotated_frame, segmented_frame_masks
    