import torch
import List, Tuple
from ..pretrained.GroundingDINO.groundingdino.util.utils import get_phrases_from_posmap
from ..pretrained.GroundingDINO.groundingdino.util.inference import predict, preprocess_caption, Model
from ..pretrained.GroundingDINO.groundingdino.models.GroundingDINO.bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)
from transformers import AutoTokenizer, BertModel, BertTokenizer

################################################################################################################
# Generate descirptors
def toy_descriptors(base_prompt):          
    '''
    toy function to mock the extending of prompts
    '''
    descriptors = ["that is on the floor","that has furry ears","that $2@zj9it8/%j",",which 8aoiU(&*YT)"]
    return descriptors

def generate_descriptors_waffle(base_prompt) ->List[str]:
    # TBD
    descriptors = []
    return descriptors

def generate_descriptors_gpt(base_prompt) ->List[str]:
    # TBD
    descriptors = []
    return descriptors

def generate_descriptors_waffle_and_gpt(base_prompt) ->List[str]:
    # TBD
    descriptors = []
    return descriptors

# And so on, TODO: fill in the functions

################################################################################################################
# Utility functions

# useful if predict_extended took a list of base_prompt
def print_dict(dict_descriptors):
    '''
    print the dict of descriptors, where
    dict_descriptors = {class:[extended captions for the class]}
    '''
    for k, v in dict_descriptors.items():
        for descriptor in v:
            print(f"  - {descriptor}")

def combine_base_and_descriptors(dict_descriptors) ->List[List[str]]:
    '''
    args: dict_descriptors = {class:[extended captions for the class]}
    returns: combined = [[class, descriptors], [class, descriptors], ...]
    '''
    combined = []
    for k, v in dict_descriptors.items():
        extended = []
        for descriptor in v:
            extended.append(f"{k}, {descriptor}")
        combined.append(extended)
    return combined

def preprocess_caption(caption: str) -> str:
    '''
    add a period at the end of the caption if it doesn't have one
    '''
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."
################################################################################################################
# adapt predict() from inference.py in GroundingDINO
def predict_extended(
    model, 
    image: torch.Tensor, 
    base_prompt: str,            # TODO(perhaps): or list of strings, e.g. ['jeans', 'bag on the table', 'book']
    box_threshold: float, 
    text_threshold: float, 
    device: str = "cuda", 
    prompt_extender: str = None
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    '''
    Adapt grounding dino inference to predict bounding boxes and phrases with optional extended captions
    Args: 
    - model, image, caption, box_threshold, text_threshold and device same as predict()
    - prompt_extender: None, 'toy', 'gpt', 'waffle', 'waflle_and_gpt', etc.
    Returns:
    - boxes: torch.Tensor of shape (N, 4) containing the bounding boxes
    - logits: torch.Tensor of shape (N, 2) containing the logits
    - phrases: List of strings containing the phrases
    '''

    classes = [base_prompt]
    descriptotors = None                   
    # Generate extended captions based on different methods
    if prompt_extender is None:
        descriptors = ['']
    elif prompt_extender == "toy":
        descriptors = toy_descriptors(base_prompt)
    # elif caption_extender == "waffle":
    #     captions = generate_descriptors_waffle(caption)
    # elif caption_extender == "gpt":
    #     captions = generate_descriptors_gpt(caption)
    # elif caption_extender == "waffle_and_gpt":
    #     captions = generate_descriptors_waffle_and_gpt(caption)
    else:
        raise ValueError(f"Unknown caption extender: {prompt_extender}")
    
    captions = [f"{base_prompt}, {descriptor}" for descriptor in descriptors]
    captions = [preprocess_caption(caption) for caption in captions]

    model = model.to(device)
    image = image.to(device)
    bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
    bert.pooler.dense.weight.requires_grad_(False)
    bert.pooler.dense.bias.requires_grad_(False)
    bert = BertModelWarper(bert_model=self.bert)
    tokenizer = BertModel.from_pretrained("bert-base-uncased")

    feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)

    with torch.no_grad():
        # outputs = model(image[None], captions=[caption]) # the model requires a batch of images and their respective captions (one per image). 
        # # for a single imgage, image[None] transforms shape from (3, H, W) to (1, 3, H, W), caption
         # encoder texts
        samples = image[None]
        max_text_len = 195
        for caption in captions:
            capt = [caption]
            
            tokenized = tokenizer(capt, padding="longest", return_tensors="pt").to(samples.device)
            special_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
            (   text_self_attention_masks,
                position_ids,
                cate_to_token_mask_list,
            ) = generate_masks_with_special_tokens_and_transfer_map(tokenized, special_tokens, tokenizer)

            if text_self_attention_masks.shape[1] > max_text_len:
                text_self_attention_masks = text_self_attention_masks[:, : max_text_len, : max_text_len]
                position_ids = position_ids[:, : max_text_len]
                tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
                tokenized["attention_mask"] = tokenized["attention_mask"][:, : max_text_len]
                tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : max_text_len]

            # extract text embeddings
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids

            bert_output = bert(**tokenized_for_encoder)  # bs, 195, 768

            encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
            text_token_mask = tokenized.attention_mask.bool()  # bs, 195
            # text_token_mask: True for nomask, False for mask
            # text_self_attention_masks: True for nomask, False for mask

            if encoded_text.shape[1] > self.max_text_len:
                encoded_text = encoded_text[:, : self.max_text_len, :]
                text_token_mask = text_token_mask[:, : self.max_text_len]
                position_ids = position_ids[:, : self.max_text_len]
                text_self_attention_masks = text_self_attention_masks[:, : self.max_text_len, : self.max_text_len]

        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }

        # import ipdb; ipdb.set_trace()

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict
        )

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        # output
        outputs_class = torch.stack(
            [layer_cls_embed(layer_hs, text_dict) for layer_cls_embed, layer_hs in zip(self.class_embed, hs)]
        )
        outputs = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    mask = prediction_logits.max(dim=1)[0] > box_threshold 
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace(".", "") for logit in logits
    ]
    
    source_h, source_w, _ = image.shape
    detections = Model.post_process_result(source_h=source_h, source_w=source_w, boxes=boxes, logits=logits)
    class_id = Model.phrases2classes(phrases=phrases, classes=classes)
    detections.class_id = class_id
    
    return final_boxes, final_logits.max(dim=1)[0], all_phrases
