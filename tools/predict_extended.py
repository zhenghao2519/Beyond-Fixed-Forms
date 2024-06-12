import torch
import copy
from typing import List, Tuple

from torch import nn
import torch.nn.functional as F
from descriptor_generator import toy_descriptors, generate_descriptors_waffle, generate_descriptors_gpt, generate_descriptors_waffle_and_gpt
from ..pretrained.GroundingDINO.groundingdino.models.GroundingDINO.transformer import Transformer, build_transformer
from ..pretrained.GroundingDINO.groundingdino.models.GroundingDINO import GroundingDINO
from ..pretrained.GroundingDINO.groundingdino.util.utils import get_phrases_from_posmap
from ..pretrained.GroundingDINO.groundingdino.util.inference import predict, preprocess_caption, Model
from ..pretrained.GroundingDINO.groundingdino.util import box_ops, get_tokenlizer
from ..pretrained.GroundingDINO.groundingdino.models.GroundingDINO.bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)
from ..pretrained.GroundingDINO.groundingdino.util.misc import (
    NestedTensor,
    inverse_sigmoid,
    nested_tensor_from_tensor_list,
)
from ..pretrained.GroundingDINO.groundingdino.models.GroundingDINO.utils import MLP, ContrastiveEmbed
from transformers import AutoTokenizer, BertModel, BertTokenizer


def predict_extended(
    model, 
    backbone: nn.Module,
    tansformer: Transformer,
    image: torch.Tensor, 
    base_prompt: str,            # TODO(perhaps): or list of strings, e.g. ['jeans', 'bag on the table', 'book']
    box_threshold: float, 
    text_threshold: float, 
    device: str = "cuda", 
    prompt_extender: str = None,
    hidden_dim: int = 256,
    max_text_len: int = 195,
    num_feature_levels: int = 1,
    two_stage_type: str = "no",
    # two stage
    dec_pred_bbox_embed_share=True,
    two_stage_class_embed_share=True,
    two_stage_bbox_embed_share=True,
    text_encoder_type: str = "bert-base-uncased",
    sub_sentence_present: bool = True,
    **kwargs
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
    # classes = [base_prompt]
    transformer = build_transformer(model.args)

    # load bert
    bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
    bert.pooler.dense.weight.requires_grad_(False)
    bert.pooler.dense.bias.requires_grad_(False)
    bert = BertModelWarper(bert_model=bert)
    tokenizer = BertModel.from_pretrained(text_encoder_type)

    feat_map = nn.Linear(bert.config.hidden_size, hidden_dim, bias=True)
    nn.init.constant_(feat_map.bias.data, 0)
    nn.init.xavier_uniform_(feat_map.weight.data)

    # special tokens
    special_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

     # prepare input projection layers
    if num_feature_levels > 1:
        num_backbone_outs = len(backbone.num_channels)
        input_proj_list = []
        for _ in range(num_backbone_outs):
            in_channels = backbone.num_channels[_]
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            )
        for _ in range(num_feature_levels - num_backbone_outs):
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            )
            in_channels = hidden_dim
        input_proj = nn.ModuleList(input_proj_list)
    else:
        assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
        input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            ]
        )
    # prepare class & box embed
    _class_embed = ContrastiveEmbed()
    _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
    nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
    nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

    if dec_pred_bbox_embed_share:
        box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
    else:
        box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)]
    class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
    bbox_embed = nn.ModuleList(box_embed_layerlist)
    class_embed = nn.ModuleList(class_embed_layerlist)
    transformer.decoder.bbox_embed = bbox_embed
    transformer.decoder.class_embed = class_embed

    # Generate extended captions based on different methods
    descriptotors = None
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
    
    # Preprocess captions
    captions = [f"{base_prompt}, {descriptor}" for descriptor in descriptors]
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
            capt = [caption] #['jeans, that has a zipper.']
        
            tokenized = tokenizer(capt, padding="longest", return_tensors="pt").to(samples.device)
            
            (   text_self_attention_masks,
                position_ids,
                cate_to_token_mask_list,
            ) = generate_masks_with_special_tokens_and_transfer_map(tokenized, special_tokens, tokenizer)

            if text_self_attention_masks.shape[1] > max_text_len:
                text_self_attention_masks = text_self_attention_masks[:, : max_text_len, : max_text_len]
                position_ids = position_ids[:, : max_text_len]
                tokenized["input_ids"] = tokenized["input_ids"][:, : max_text_len]
                tokenized["attention_mask"] = tokenized["attention_mask"][:, : max_text_len]
                tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : max_text_len]

                # extract text embeddings
            if sub_sentence_present:
                tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
                tokenized_for_encoder["attention_mask"] = text_self_attention_masks
                tokenized_for_encoder["position_ids"] = position_ids
            else:
                # import ipdb; ipdb.set_trace()
                tokenized_for_encoder = tokenized

                bert_output = bert(**tokenized_for_encoder)                # bs, 195, 768

                encoded_text = feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
                text_token_mask = tokenized.attention_mask.bool()  # bs, 195
                # text_token_mask: True for nomask, False for mask
                # text_self_attention_masks: True for nomask, False for mask

                if encoded_text.shape[1] > max_text_len:
                    encoded_text = encoded_text[:, : max_text_len, :]
                    text_token_mask = text_token_mask[:, : max_text_len]
                    position_ids = position_ids[:, : max_text_len]
                    text_self_attention_masks = text_self_attention_masks[:, : max_text_len, : max_text_len]

            all_encoded_text.append(encoded_text)
            all_text_token_mask.append(text_token_mask)
            all_position_ids.append(position_ids)
            all_text_self_attention_masks.append(text_self_attention_masks)

        # Stack all the encoded texts, masks, etc.
        encoded_text = torch.cat(all_encoded_text, dim=0)
        text_token_mask = torch.cat(all_text_token_mask, dim=0)
        position_ids = torch.cat(all_position_ids, dim=0)
        text_self_attention_masks = torch.cat(all_text_self_attention_masks, dim=0)

        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }

        # import ipdb; ipdb.set_trace()

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, num_feature_levels):
                if l == _len_srcs:
                    src = input_proj[l](features[-1].tensors)
                else:
                    src = input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = GroundingDINO.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict
        )

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        # output
        outputs_class = torch.stack(
            [layer_cls_embed(layer_hs, text_dict) for layer_cls_embed, layer_hs in zip(class_embed, hs)]
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
    
    # class_id = Model.phrases2classes(phrases=phrases, classes=classes)  # for one class this is not needed

    return boxes, logits.max(dim=1)[0], phrases
