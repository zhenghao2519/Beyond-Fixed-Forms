import torch
import copy
from typing import List, Tuple
import numpy as np

from torch import nn
import torch.nn.functional as F
from descriptor_generator import toy_descriptors, generate_descriptors_waffle, generate_descriptors_gpt, generate_descriptors_waffle_and_gpt
from groundingdino.models.GroundingDINO.transformer import Transformer, build_transformer
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.inference import predict, preprocess_caption, Model
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.models.GroundingDINO.bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)
from groundingdino.util.misc import (
    NestedTensor,
    inverse_sigmoid,
    nested_tensor_from_tensor_list,
)
from groundingdino.models.GroundingDINO.utils import MLP, ContrastiveEmbed
from transformers import AutoTokenizer, BertModel, BertTokenizer


def predict_extended(
    model, 
    # backbone: nn.Module,
    # tansformer: Transformer,
    image: torch.Tensor, 
    base_prompt: str,  # only one object # TODO(perhaps): or list of strings, e.g. ['jeans', 'bag on the table', 'book']
    box_threshold: float, 
    text_threshold: float, 
    device: str = "cuda", 
    prompt_extender: str = None,
    # hidden_dim: int = 256,
    # max_text_len: int = 195,
    # num_feature_levels: int = 1,
    # two_stage_type: str = "no",
    # # two stage
    # dec_pred_bbox_embed_share=True,
    # two_stage_class_embed_share=True,
    # two_stage_bbox_embed_share=True,
    # text_encoder_type: str = "bert-base-uncased",
    # sub_sentence_present: bool = True,
    # **kwargs
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
    # # classes = [base_prompt]
    # transformer = build_transformer(model.args)

    # Generate extended captions based on different methods
    descriptors = None
    if prompt_extender is None:
        descriptors_structured = ['']
    elif prompt_extender == "toy":
        descriptors_structured = toy_descriptors(base_prompt)
    # elif caption_extender == "waffle":
    #     captions = generate_descriptors_waffle(caption)
    # elif caption_extender == "gpt":
    #     captions = generate_descriptors_gpt(caption)
    # elif caption_extender == "waffle_and_gpt":
    #     captions = generate_descriptors_waffle_and_gpt(caption)
    else:
        raise ValueError(f"Unknown caption extender: {prompt_extender}")
    
    # Preprocess captions
    captions = descriptors_structured
    # print("Captions: ", captions)
    captions = [preprocess_caption(caption) for caption in captions]
    
    model = model.to(device)
    image = image.to(device)

    all_encoded_text = []
    all_text_token_mask = []
    all_position_ids = []
    all_text_self_attention_masks = []

    # FORWARD PASS THROUGH THE GROUNDING DINO
    with torch.no_grad():
        # outputs = model(image[None], captions=[caption])
        samples = image[None]
        captions = captions

        # encoder texts
        for caption in captions:
            capt = [caption] #e.g., ['jeans, which has a zipper.']
        
            tokenized = model.tokenizer(capt, padding="max_length", return_tensors="pt").to(samples.device)
            
            (   text_self_attention_masks,
                position_ids,
                cate_to_token_mask_list,
            ) = generate_masks_with_special_tokens_and_transfer_map(tokenized, model.specical_tokens, model.tokenizer)

            if text_self_attention_masks.shape[1] > model.max_text_len:
                text_self_attention_masks = text_self_attention_masks[:, : model.max_text_len, : model.max_text_len]
                position_ids = position_ids[:, : model.max_text_len]
                tokenized["input_ids"] = tokenized["input_ids"][:, : model.max_text_len]
                tokenized["attention_mask"] = tokenized["attention_mask"][:, : model.max_text_len]
                tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : model.max_text_len]

                # extract text embeddings
            if model.sub_sentence_present:
                tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
                tokenized_for_encoder["attention_mask"] = text_self_attention_masks
                tokenized_for_encoder["position_ids"] = position_ids
            else:
                # import ipdb; ipdb.set_trace()
                tokenized_for_encoder = tokenized

            bert_output = model.bert(**tokenized_for_encoder)                # bs, 195, 768

            encoded_text = model.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
            text_token_mask = tokenized.attention_mask.bool()  # bs, 195
            # text_token_mask: True for nomask, False for mask
            # text_self_attention_masks: True for nomask, False for mask
            # print("====================DEBUG" , encoded_text.shape, text_token_mask.shape, position_ids.shape, text_self_attention_masks.shape)

            if encoded_text.shape[1] > model.max_text_len:
                encoded_text = encoded_text[:, : model.max_text_len, :]
                text_token_mask = text_token_mask[:, : model.max_text_len]
                position_ids = position_ids[:, : model.max_text_len]
                text_self_attention_masks = text_self_attention_masks[:, : model.max_text_len, : model.max_text_len]

            all_encoded_text.append(encoded_text)
            all_text_token_mask.append(text_token_mask)
            all_position_ids.append(position_ids)
            all_text_self_attention_masks.append(text_self_attention_masks)
            # print(text_self_attention_masks)
            

        # Stack all the encoded texts, masks, etc.
        encoded_text = torch.cat(all_encoded_text, dim=0).mean(dim=0, keepdim=True)
        text_token_mask = torch.cat(all_text_token_mask, dim=0).max(dim=0, keepdim = True).values
        position_ids = torch.cat(all_position_ids, dim=0).max(dim=0, keepdim = True).values
        text_self_attention_masks = torch.cat(all_text_self_attention_masks, dim=0).max(dim=0, keepdim = True).values

        # print("==========outer==========DEBUG" , encoded_text.shape, text_token_mask.shape, position_ids.shape, text_self_attention_masks.shape)
        # print(text_token_mask, position_ids)

        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }

        # import ipdb; ipdb.set_trace()

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = model.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(model.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if model.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, model.num_feature_levels):
                if l == _len_srcs:
                    src = model.input_proj[l](features[-1].tensors)
                else:
                    src = model.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = model.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = model.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict
        )

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], model.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        # output
        outputs_class = torch.stack(
            [layer_cls_embed(layer_hs, text_dict) for layer_cls_embed, layer_hs in zip(model.class_embed, hs)]
        )
        outputs = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    mask = prediction_logits.max(dim=1)[0] > box_threshold 
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    tokenizer = model.tokenizer
    tokenized = tokenizer(base_prompt) # Clothes
    # for predict_classes : check if predicted phrase is with in the classes
    # Model.predict_with_classes(phrases=phrases, classes=classes)
    

    phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace(".", "") for logit in logits
    ]
    
    # class_id = Model.phrases2classes(phrases=phrases, classes=classes)  # for one class this is not needed

    return boxes, logits.max(dim=1)[0], phrases


@staticmethod
def phrases2classes(phrases: List[str], classes: List[str]) -> np.ndarray:
    class_ids = []
    for phrase in phrases:
        try:
            class_ids.append(classes.index(phrase))
        except ValueError:
            class_ids.append(None)
    return np.array(class_ids)
