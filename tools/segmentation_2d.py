#%% IMPORTS
# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict
from torch.utils.data import DataLoader

# segment anything
from segment_anything import build_sam, SamPredictor

# tools
import descriptor_generator
# from predict_extended import predict_extended

# other imports
from torch.nn import functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np
from PIL import Image
import os
import torch
import clip
from tqdm import tqdm
from termcolor import colored
from collections import OrderedDict
import argparse
import yaml
from huggingface_hub import hf_hub_download
from munch import Munch
import requests

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# Device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

#%% LOAD MODELS
def download_file(url, filename):
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Open the file in binary write mode and save the content to the file
        with open(filename, "wb") as file:
            file.write(response.content)
        print(f"File downloaded successfully: {filename}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")


def load_model_hf(repo_id, filename, ckpt_config_filename, device=device):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    args.device = device
    print(f"Loading hf model with args: {args}")
    model = build_model(args)

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    # print("cache file",cache_file)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


def load_grounded_sam():
    # load grounding dino
    ckpt_repo_id = cfg.ckpt_repo_id
    ckpt_filename = cfg.ckpt_filename
    ckpt_config_filename = cfg.ckpt_config_filename
    groundingdino_model = load_model_hf(
        ckpt_repo_id, ckpt_filename, ckpt_config_filename, device
    )
    # load sam
    sam_checkpoint = (
        cfg.sam_checkpoint
    )  # wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    if not os.path.exists(sam_checkpoint):
        url = cfg.sam_url
        filename = cfg.sam_checkpoint
        download_file(url, filename)
    sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    print(colored("GroundingDINO & SAM loaded", "yellow", attrs=["bold"]))
    return groundingdino_model, sam_predictor

def load_clip_model(model_size='ViT-L/14'):
    model, _ = clip.load(model_size, device=device, jit=False)
    model.eval()
    model.requires_grad_(False)
    return model

#%% GROUNDED SAM
# Detect object using grounding DINO
def detect(
    image_source,
    image,
    capt_feature_ensembled,
    text_prompt,
    model,
    clip_model,
    box_threshold=0.4,
    text_threshold=0.4,
    device="cuda",
    filter_with_clip_feature=False,
    clip_size=None,
    similarity_threshold=None,
):
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )
    # remove boxes with labels different from the base prompt
    if len(phrases) != 0:
        # print(colored(f"Detected phrases: {phrases}", "green", attrs=["bold"]))
        # print(text_prompt)
        indices = [i for i, phrase in enumerate(phrases) if text_prompt in phrase]
        # print("box_indices", indices)
        boxes = boxes[indices]
        logits = logits[indices]
        phrases = [phrases[i] for i in indices]
        # print("after filtering", phrases)
        

    if filter_with_clip_feature:
        if clip_size is None:
            raise ValueError("Please specify the CLIP model size for filtering")
        if similarity_threshold is None:
            raise ValueError("Please specify the similarity threshold for filtering")
        if boxes is None or len(boxes) == 0:
            return image_source, boxes, logits, phrases
        else:
            
            # print(colored(f"Modle {clip_size} loaded", "yellow", attrs=["bold"]))
            # print(colored(f"Caption feature ensembled, shape: {capt_feature_ensembled.shape}", "green", attrs=["bold"]))
            boxes, logits, phrases = bbox_filter(image, boxes, phrases, capt_feature_ensembled, clip_threshold=similarity_threshold, clip_model=clip_model)
    
    # Ensure logits is a tensor before squeezing
    if isinstance(logits, torch.Tensor):
        logits = logits.squeeze(1) if logits.ndim > 1 else logits
    # Convert logits to list of floats for annotation
    logits_list = logits.tolist() if isinstance(logits, torch.Tensor) else logits

    annotated_frame = annotate(
        image_source=image_source, boxes=boxes, logits=logits_list, phrases=phrases
    )
    annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB
    # print(f"Detected {len(boxes)} boxes")
    return annotated_frame, boxes, logits, phrases  # box, confidence, label


# Segment and draw masks using SAM
def segment(image, sam_model, boxes):
    if len(boxes) == 0:
        print("No boxes detected")
        return None
    sam_model.set_image(image)
    H, W, _ = image.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = sam_model.transform.apply_boxes_torch(
        boxes_xyxy.to(device), image.shape[:2]
    )  # boxes in xyxy format,
    masks, _, _ = sam_model.predict_torch(  # masks in [0, 1] range
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,  # if True, returns masks for each box
    )
    if masks is None:
        print("No masks generated")
    return masks.cpu() if masks is not None else None


def draw_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert(
        "RGBA"
    )  # annotated fram in PIL format
    mask_image_pil = Image.fromarray(
        (mask_image.cpu().numpy() * 255).astype(np.uint8)
    ).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

# Batch inference using Grounded SAM and save results
def inference_grounded_sam(
    groundingdino_model,
    sam_predictor,
    clip_model,
    scene_id,
    image_paths,
    base_prompt,
    dino_box_threshold,
    dino_text_threshold,
    mask_2d_dir,
    draw=True,
    filter_with_clip_feature=False,
    clip_size=None,
    similarity_threshold=None,
):
    """Inference using Grounding DINO and SAM
    Args:
        image_path (_type_): path of the image
        base_prompt (_type_): text query
    Returns:
        annotated_frame: frame with detected boxes from grounding DINO
        detected_boxes: detected boxes from grounding DINO
        segmented_frame_masks: masks from SAM
    """

    results = []
    num_img_with_boxes = 0
    
    if filter_with_clip_feature:
        capt_feature_ensembled = compute_avg_description_encodings(text_prompt, clip_model, mode='waffle')
                    
    for i, image_path in enumerate(
        tqdm(image_paths, desc="Processing images", leave=False)
    ):
        frame_id = image_path.split("/")[-1]  # get frame id from image path
        # print(frame_id)
        image_source, image = load_image(image_path)
        if image_source is None or image is None:  # skip the image if not loaded
            continue
        annotated_frame, detected_boxes, confidences, labels = detect(
            image_source,
            image,
            capt_feature_ensembled,
            text_prompt=base_prompt,
            model=groundingdino_model.to(device),
            box_threshold=dino_box_threshold,
            text_threshold=dino_text_threshold,
            filter_with_clip_feature=filter_with_clip_feature,
            clip_model = clip_model,
            clip_size=clip_size,
            similarity_threshold=similarity_threshold,
        )
        if (
            detected_boxes is None or len(detected_boxes) == 0
        ):  # skip the image if no boxes detected
            continue
        num_img_with_boxes += 1
        segmented_frame_masks = segment(
            image_source, sam_predictor, boxes=detected_boxes
        )

        if draw:
            annotated_frame_with_mask = annotated_frame
            for i in range(segmented_frame_masks.shape[0]):
                annotated_frame_with_mask = draw_mask(
                    segmented_frame_masks[i][0], annotated_frame_with_mask
                )

            # if not os.ath.exists(os.path.join(mask_2d_dir, "vis_masks")):
            os.makedirs(
                os.path.join(mask_2d_dir, "vis_masks", base_prompt, scene_id,), exist_ok=True
            )
            Image.fromarray(annotated_frame_with_mask).save(
                os.path.join(
                    mask_2d_dir, "vis_masks", base_prompt, scene_id,frame_id[:-4] + ".png"
                )
            )

        results.append(
            {
                "frame_id": frame_id,
                "segmented_frame_masks": segmented_frame_masks.to(torch.bool),  # (M, 1, W, H)
                "confidences": confidences,
                "labels": labels,
               
            }
        )  # add keys "annotated_frame", "detected_boxes" if needed

    print(f"Number of images with at least one box detected: {num_img_with_boxes}")
    return results


#%% FILTER DINO OUTPUTS USING CAPTION FEATURE
def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
def compute_avg_description_encodings(base_prompt: str, clip_model, mode: str='waffle'):
    '''
    Extend given base_prompt using given extending method,
    encode the extended prompts with CLIP model,
    then compute the average description encodings
    '''
    extended_descriptions = descriptor_generator.descr_generator_selector(base_prompt, method=mode)
    description_encodings = OrderedDict()
    # for k, v in tqdm(extended_descriptions.items(), total=len(extended_descriptions), desc='Encoding Descriptions...'):
    for k, v in extended_descriptions.items():
        description_encodings[k] = F.normalize(clip_model.encode_text(clip.tokenize(v).to(device)))
    descr_means = torch.cat([x.mean(dim=0).reshape(1, -1) for x in description_encodings.values()])
    descr_means /= descr_means.norm(dim=-1, keepdim=True)
    return descr_means # shape: (n_classes, 512)


def bbox_filter(image, boxes, phrases, capt_feature_ensembled, clip_threshold=0.5, clip_model= None):
    '''
    Compute image embeddings of boxed areas, and filter out 
    boxes whose similarity with the given precomputed text caption feature is below a threshold.
    - args:
        image: torch.Tensor, shape: (H, W, 3), the transformed image tensor
        boxes: torch.Tensor, shape: (n_boxes, 4), the detected boxes in (cx, cy, w, h) format
        capt_feature_ensembled: torch.Tensor, shape: (n_classes, 512), the precomputed text caption feature
        clip_threshold: similarity threshold
        clip_model: CLIP model
    - returns:
        boxes_filtered: torch.Tensor, shape: (n_filtered_boxes, 4)
    '''
    
    if boxes is None or len(boxes) == 0:    # No boxes to filter
        return boxes, [], []
    preprocess = _transform(224)  # Using 224 as the typical input size for ViT-B/32
    
    # Extract and resize box regions
    _, H, W = image.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([W, H, W, H])
    # print(f"boxes_xyxy: {boxes_xyxy}")
    box_regions = []
    for i, box in enumerate(boxes_xyxy):
        box[0] = max(0, box[0])
        box[1] = max(0, box[1])
        box[2] = min(W, box[2])
        box[3] = min(H, box[3])
        box_region = image[:, int(box[1]):int(box[3]), int(box[0]):int(box[2])] # shape: (3, H_cropped, W_cropped) torch.float32

        box_region = Image.fromarray((box_region.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) #  permute to (H,W,3), * 255 because PIL expects 0-255
        # For visualization:
        # os.makedirs(f"/home/jie_zhenghao/Beyond-Fixed-Forms/output/tempt", exist_ok=True)
        # save_path = f"/home/jie_zhenghao/Beyond-Fixed-Forms/output/tempt/box_region{i}.jpg" 
        # box_region.save(save_path)
        box_region = preprocess(box_region).unsqueeze(0).to(device)
        # For visualization:
        # save_path = f"/home/jie_zhenghao/Beyond-Fixed-Forms/output/tempt/box_region{i}_preprocessed.jpg"
        # Image.fromarray((box_region.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(save_path)
        box_regions.append(box_region)
    
    # Compute image embeddings for each preprocessed box region
    box_embeddings = []
    for region in box_regions:
        # region_tensor = region.unsqueeze(0).to(device)  # shape: (1, 3, H_cropped, W_cropped)
        # print(f"region_tensor.shape: {region_tensor.shape}")
        with torch.no_grad():
            region_embedding = clip_model.encode_image(region) # shape: (1, 512)
            box_embeddings.append(F.normalize(region_embedding))
    box_embeddings = torch.cat(box_embeddings, dim=0) # shape: (n_boxes, 512)
    # print(colored(f"box_embeddings.shape: {box_embeddings.shape}", "green", attrs=["bold"]))

    # Compute similarity score
    box_similarities = box_embeddings @ capt_feature_ensembled.T
    print(colored(f"box_similarities: {box_similarities}", "green", attrs=["bold"]))

    # Filter boxes
    mask = (box_similarities >= clip_threshold).squeeze(1)
    boxes_filtered = boxes[mask]
    logits_filtered = box_similarities[mask]
    phrases_filtered = [phrases[i] for i in range(len(mask)) if mask[i]]

    return boxes_filtered, logits_filtered, phrases_filtered


def get_parser():
    parser = argparse.ArgumentParser(description="Configuration Beyond-Fixed-Forms")
    parser.add_argument("--config", type=str, required=True, help="Config, specify the path to config.yaml file")
    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()
    cfg = Munch.fromDict(
        yaml.safe_load(open(args.config, "r").read())
    )  # Can run in command line using "python your_script.py --config config.yaml"
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    dino_box_threshold = cfg.dino_box_threshold
    dino_text_threshold = cfg.dino_text_threshold
    filter_with_clip_feature = cfg.filter_with_CLIP_feature
    clip_size=cfg.CLIP_model_size
    similarity_threshold=cfg.similarity_threshold
    text_prompt = cfg.base_prompt
    mask_2d_dir = cfg.mask_2d_dir # 2d output path
    
    scene_2d_dir = cfg.scene_2d_dir
    # scenes = ['scene0011_00', 'scene0011_01', 'scene0015_00', 'scene0019_00', 'scene0019_01', 'scene0025_00', 'scene0025_01', 'scene0025_02', 'scene0030_00', 'scene0030_01', 'scene0030_02', 'scene0046_00', 'scene0046_01', 'scene0046_02', 'scene0050_00', 'scene0050_01', 'scene0050_02', 'scene0063_00', 'scene0064_00', 'scene0064_01', 'scene0077_00', 'scene0077_01', 'scene0081_00', 'scene0081_01', 'scene0081_02', 'scene0084_00', 'scene0084_01', 'scene0084_02', 'scene0086_00', 'scene0086_01', 'scene0086_02', 'scene0088_00', 'scene0088_01', 'scene0088_02', 'scene0088_03', 'scene0095_00', 'scene0095_01', 'scene0100_00', 'scene0100_01', 'scene0100_02', 'scene0131_00', 'scene0131_01', 'scene0131_02', 'scene0139_00', 'scene0144_00', 'scene0144_01', 'scene0146_00', 'scene0146_01', 'scene0146_02', 'scene0149_00', 'scene0153_00', 'scene0153_01', 'scene0164_00', 'scene0164_01', 'scene0164_02', 'scene0164_03', 'scene0169_00', 'scene0169_01', 'scene0187_00', 'scene0187_01', 'scene0193_00', 'scene0193_01', 'scene0196_00', 'scene0203_00', 'scene0203_01', 'scene0203_02', 'scene0207_00', 'scene0207_01', 'scene0207_02', 'scene0208_00', 'scene0217_00', 'scene0221_00', 'scene0221_01', 'scene0222_00', 'scene0222_01', 'scene0231_00', 'scene0231_01', 'scene0231_02', 'scene0246_00', 'scene0249_00', 'scene0251_00', 'scene0256_00', 'scene0256_01', 'scene0256_02', 'scene0257_00', 'scene0277_00', 'scene0277_01', 'scene0277_02', 'scene0278_00', 'scene0278_01', 'scene0300_00', 'scene0300_01', 'scene0304_00', 'scene0307_00', 'scene0307_01', 'scene0307_02', 'scene0314_00', 'scene0316_00', 'scene0328_00', 'scene0329_00', 'scene0329_01', 'scene0329_02', 'scene0334_00', 'scene0334_01', 'scene0334_02', 'scene0338_00', 'scene0338_01', 'scene0338_02', 'scene0342_00', 'scene0343_00', 'scene0351_00', 'scene0351_01', 'scene0353_00', 'scene0353_01', 'scene0353_02', 'scene0354_00', 'scene0355_00', 'scene0355_01', 'scene0356_00', 'scene0356_01', 'scene0356_02', 'scene0357_00', 'scene0357_01', 'scene0377_00', 'scene0377_01', 'scene0377_02', 'scene0378_00', 'scene0378_01', 'scene0378_02', 'scene0382_00', 'scene0382_01', 'scene0389_00', 'scene0406_00', 'scene0406_01', 'scene0406_02', 'scene0412_00', 'scene0412_01', 'scene0414_00', 'scene0423_00', 'scene0423_01', 'scene0423_02', 'scene0426_00', 'scene0426_01', 'scene0426_02', 'scene0426_03', 'scene0427_00', 'scene0430_00', 'scene0430_01', 'scene0432_00', 'scene0432_01', 'scene0435_00', 'scene0435_01', 'scene0435_02', 'scene0435_03', 'scene0441_00', 'scene0458_00', 'scene0458_01', 'scene0461_00', 'scene0462_00', 'scene0474_00', 'scene0474_01', 'scene0474_02', 'scene0474_03', 'scene0474_04', 'scene0474_05', 'scene0488_00', 'scene0488_01', 'scene0490_00', 'scene0494_00', 'scene0496_00', 'scene0500_00', 'scene0500_01', 'scene0518_00', 'scene0527_00', 'scene0535_00', 'scene0549_00', 'scene0549_01', 'scene0550_00', 'scene0552_00', 'scene0552_01', 'scene0553_00', 'scene0553_01', 'scene0553_02', 'scene0558_00', 'scene0558_01', 'scene0558_02', 'scene0559_00', 'scene0559_01', 'scene0559_02', 'scene0565_00', 'scene0568_00', 'scene0568_01', 'scene0568_02', 'scene0574_00', 'scene0574_01', 'scene0574_02', 'scene0575_00', 'scene0575_01', 'scene0575_02', 'scene0578_00', 'scene0578_01', 'scene0578_02', 'scene0580_00', 'scene0580_01', 'scene0583_00', 'scene0583_01', 'scene0583_02', 'scene0591_00', 'scene0591_01', 'scene0591_02', 'scene0593_00', 'scene0593_01', 'scene0595_00', 'scene0598_00', 'scene0598_01', 'scene0598_02', 'scene0599_00', 'scene0599_01', 'scene0599_02', 'scene0606_00', 'scene0606_01', 'scene0606_02', 'scene0607_00', 'scene0607_01', 'scene0608_00', 'scene0608_01', 'scene0608_02', 'scene0609_00', 'scene0609_01', 'scene0609_02', 'scene0609_03', 'scene0616_00', 'scene0616_01', 'scene0618_00', 'scene0621_00', 'scene0629_00', 'scene0629_01', 'scene0629_02', 'scene0633_00', 'scene0633_01', 'scene0643_00', 'scene0644_00', 'scene0645_00', 'scene0645_01', 'scene0645_02', 'scene0647_00', 'scene0647_01', 'scene0648_00', 'scene0648_01', 'scene0651_00', 'scene0651_01', 'scene0651_02', 'scene0652_00', 'scene0653_00', 'scene0653_01', 'scene0655_00', 'scene0655_01', 'scene0655_02', 'scene0658_00', 'scene0660_00', 'scene0663_00', 'scene0663_01', 'scene0663_02', 'scene0664_00', 'scene0664_01', 'scene0664_02', 'scene0665_00', 'scene0665_01', 'scene0670_00', 'scene0670_01', 'scene0671_00', 'scene0671_01', 'scene0678_00', 'scene0678_01', 'scene0678_02', 'scene0684_00', 'scene0684_01', 'scene0685_00', 'scene0685_01', 'scene0685_02', 'scene0686_00', 'scene0686_01', 'scene0686_02', 'scene0689_00', 'scene0690_00', 'scene0690_01', 'scene0693_00', 'scene0693_01', 'scene0693_02', 'scene0695_00', 'scene0695_01', 'scene0695_02', 'scene0695_03', 'scene0696_00', 'scene0696_01', 'scene0696_02', 'scene0697_00', 'scene0697_01', 'scene0697_02', 'scene0697_03', 'scene0699_00', 'scene0700_00', 'scene0700_01', 'scene0700_02', 'scene0701_00', 'scene0701_01', 'scene0701_02', 'scene0702_00', 'scene0702_01', 'scene0702_02', 'scene0704_00', 'scene0704_01']
    # scenes = sorted([s for s in scenes if s.endswith("_00")])[70:80]
    scenes = ['scene0435_00']
    print("Number of scenes:", len(scenes))
    
    groundingdino_model, sam_predictor = load_grounded_sam()
    clip_model = load_clip_model(clip_size)
    
    for scene_id in tqdm(scenes):
        print("Working on ", scene_id)
        image_dir = os.path.join(scene_2d_dir, scene_id, "color")
        image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
        image_files.sort(
            key=lambda x: int(x.split(".")[0])
        )  # sort numerically, 1.jpg, 2.jpg, 3.jpg ...
        downsampled_image_files = image_files[::cfg.downsample_ratio]  # get one image every 10 frames
        print(f"Number of downsampled_images:{len(downsampled_image_files)}")
        downsampled_images_paths = [
            os.path.join(image_dir, f) for f in downsampled_image_files
        ]

        
        with torch.cuda.amp.autocast():
            grounded_sam_results = inference_grounded_sam(
                groundingdino_model,
                sam_predictor,
                clip_model,
                scene_id,
                downsampled_images_paths,
                text_prompt,
                dino_box_threshold,
                dino_text_threshold,
                mask_2d_dir,
                filter_with_clip_feature=filter_with_clip_feature,
                clip_size=clip_size,
                similarity_threshold=similarity_threshold,
                draw=True,
            )
            os.makedirs(os.path.join(mask_2d_dir, text_prompt), exist_ok=True)
            mask_2d_path = os.path.join(mask_2d_dir, text_prompt, f"{scene_id}.pth")
            torch.save(
                grounded_sam_results, mask_2d_path
            )  # save all segmented frame masks in a file
            torch.cuda.empty_cache()
