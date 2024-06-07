def get_text_embedding(model, prompt):  # helper function to get text embedding
     # Ensure the prompt is a single string
    if isinstance(prompt, str):
        prompt = [prompt]
    
    # Tokenize the prompt
    tokenized = model.tokenizer(prompt, padding="longest", return_tensors="pt").to(device)
    (
        text_self_attention_masks,
        position_ids,
        cate_to_token_mask_list,
    ) = model.generate_masks_with_special_tokens_and_transfer_map(
        tokenized, model.specical_tokens, model.tokenizer
    )
    
    # Trim text to max length
    if text_self_attention_masks.shape[1] > model.max_text_len:
        text_self_attention_masks = text_self_attention_masks[:, :model.max_text_len, :model.max_text_len]
        position_ids = position_ids[:, :model.max_text_len]
        tokenized["input_ids"] = tokenized["input_ids"][:, :model.max_text_len]
        tokenized["attention_mask"] = tokenized["attention_mask"][:, :model.max_text_len]
        tokenized["token_type_ids"] = tokenized["token_type_ids"][:, :model.max_text_len]
    
    # Prepare input for BERT
    if model.sub_sentence_present:
        tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
        tokenized_for_encoder["attention_mask"] = text_self_attention_masks
        tokenized_for_encoder["position_ids"] = position_ids
    else:
        tokenized_for_encoder = tokenized
    
    # Get BERT output
    bert_output = model.bert(**tokenized_for_encoder)
    encoded_text = model.feat_map(bert_output["last_hidden_state"])
    text_token_mask = tokenized.attention_mask.bool()

    if encoded_text.shape[1] > model.max_text_len:           #trim the text to the maximum text length, 256 here
        encoded_text = encoded_text[:, : model.max_text_len, :]
        text_token_mask = text_token_mask[:, : model.max_text_len]
        position_ids = position_ids[:, : model.max_text_len]
        text_self_attention_masks = text_self_attention_masks[
            :, : model.max_text_len, : model.max_text_len
        ]
    
    return encoded_text, text_token_mask




def predict_with_averaged_embeddings(
        model, 
        image: torch.Tensor, 
        averaged_embedding: torch.Tensor,
        combined_text_token_mask: torch.Tensor, 
        box_threshold: float, 
        text_threshold: float, 
        device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        
    with torch.no_grad():
        outputs = model(image[None], averaged_embedding=averaged_embedding, combined_text_token_mask=combined_text_token_mask)  # Pass the embedding directly

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    phrases = ["detected object"] * len(boxes)  # Simplified for this example

    return boxes, logits.max(dim=1)[0], phrases


    # Tokenize and process each caption to get embeddings
#     processed_prompts = [model.tokenizer(caption, padding="longest", return_tensors="pt").to(device) for caption in prompts]
#     embeddings = [model.bert(caption)['last_hidden_state'].mean(dim=1) for caption in processed_prompts]  # Get the mean embedding per caption
#     avg_embeddings = torch.mean(torch.stack(embeddings), dim=0)  # Average the embeddings across extended prompts


def detect_plus(image, text_prompts, model, box_threshold = 0.2, text_threshold = 0.25):

    embeddings = []
    text_token_masks = []
    for prompt in text_prompts:
        prompt_embedding, text_token_mask = get_text_embedding(model, prompt)
        embeddings.append(prompt_embedding)
        text_token_masks.append(text_token_mask)

    avg_embedding = torch.mean(torch.stack(embeddings), dim=0)
    combined_text_token_mask = torch.any(torch.stack(text_token_masks), dim=0)

    boxes, logits, phrases = predict_with_averaged_embeddings(
        model=model,
        image=image,
        avg_embedding=avg_embedding,
        combined_text_token_mask=combined_text_token_mask,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device
    )
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB for display
    return annotated_frame, boxes

annotated_frame, detected_boxes = detect_plus(image, text_prompts=extended_prompts, model=groundingdino_model.to(device))
Image.fromarray(annotated_frame)