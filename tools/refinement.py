import numpy as np

# import open3d as o3d
import yaml

import matplotlib.pyplot as plt
import torch
import os
import argparse

from configs import config as cfg
from munch import Munch

from tqdm import tqdm

import clip


"""
1. Process Stage 1 masks
"""


def rle_decode(rle):
    """
    Open3DIS as stage1, "ins" is RLE format
    """
    length = rle["length"]
    s = rle["counts"]

    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask

def scene_checkpoint_file(class_name):
    # return f"refinement_checkpoint_{class_name}.yaml"
    return f"./checkpoints/refinement_checkpoint_{class_name}.yaml"

def read_scene_checkpoint(class_name):
    checkpoint_file = scene_checkpoint_file(class_name)
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as file:
            return yaml.safe_load(file)
    return {}

def write_scene_checkpoint(class_name, checkpoint):
    checkpoint_file = scene_checkpoint_file(class_name)
    with open(checkpoint_file, 'w') as file:
        yaml.safe_dump(checkpoint, file)


def idx_to_label(idx):
    SCANNET200 = "chair.table.door.couch.cabinet.shelf.desk.office_chair.bed.pillow.sink.picture.window.toilet.bookshelf.monitor.curtain.book.armchair.coffee_table.box.refrigerator.lamp.kitchen_cabinet.towel.clothes.tv.nightstand.counter.dresser.stool.cushion.plant.ceiling.bathtub.end_table.dining_table.keyboard.bag.backpack.toilet_paper.printer.tv_stand.whiteboard.blanket.shower_curtain.trash_can.closet.stairs.microwave.stove.shoe.computer_tower.bottle.bin.ottoman.bench.board.washing_machine.mirror.copier.basket.sofa_chair.file_cabinet.fan.laptop.shower.paper.person.paper_towel_dispenser.oven.blinds.rack.plate.blackboard.piano.suitcase.rail.radiator.recycling_bin.container.wardrobe.soap_dispenser.telephone.bucket.clock.stand.light.laundry_basket.pipe.clothes_dryer.guitar.toilet_paper_holder.seat.speaker.column.bicycle.ladder.bathroom_stall.shower_wall.cup.jacket.storage_bin.coffee_maker.dishwasher.paper_towel_roll.machine.mat.windowsill.bar.toaster.bulletin_board.ironing_board.fireplace.soap_dish.kitchen_counter.doorframe.toilet_paper_dispenser.mini_fridge.fire_extinguisher.ball.hat.shower_curtain_rod.water_cooler.paper_cutter.tray.shower_door.pillar.ledge.toaster_oven.mouse.toilet_seat_cover_dispenser.furniture.cart.storage_container.scale.tissue_box.light_switch.crate.power_outlet.decoration.sign.projector.closet_door.vacuum_cleaner.candle.plunger.stuffed_animal.headphones.dish_rack.broom.guitar_case.range_hood.dustpan.hair_dryer.water_bottle.handicap_bar.purse.vent.shower_floor.water_pitcher.mailbox.bowl.paper_bag.alarm_clock.music_stand.projector_screen.divider.laundry_detergent.bathroom_counter.object.bathroom_vanity.closet_wall.laundry_hamper.bathroom_stall_door.ceiling_light.trash_bin.dumbbell.stair_rail.tube.bathroom_cabinet.cd_case.closet_rod.coffee_kettle.structure.shower_head.keyboard_piano.case_of_water_bottles.coat_rack.storage_organizer.folded_chair.fire_alarm.power_strip.calendar.poster.potted_plant.luggage.mattress"
    # SCANNETV2 = 'cabinet.bed.chair.sofa.table.door.window.bookshelf.picture.counter.desk.curtain.refrigerator.shower_curtain.toilet.sink.bathtub'
    class_names = SCANNET200.split(".")
    return class_names[idx]

"""
2. Find matched masks between stages
"""


def calculate_iou_between_stages(
    mask_1: torch.Tensor, mask_2: torch.Tensor
) -> torch.Tensor:
    """calculate iou between all masks in mask_1 and mask_2

    args:
        mask_1: torch.Tensor, shape=(n, x)
        mask_2: torch.Tensor, shape=(m, x)

    returns:
        torch.Tensor, shape=(m, n)
    """

    mask_1 = mask_1.float()
    mask_2 = mask_2.float()
    intersection = mask_1 @ mask_2.T
    union = (
        mask_1.sum(dim=-1, keepdim=True)
        + mask_2.sum(dim=-1, keepdim=True).T
        - intersection
    )
    return (intersection / union).T


def compute_clip_similarity(model, text1, text2):
    """compute similarity between text1 and text2 using clip model
    args:
        model: clip model
        text1: str
        text2: str

    returns:
        float: similarity between text1 and text2
    """
    text1 = clip.tokenize([text1]).to(device)
    text2 = clip.tokenize([text2]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text1)
        text_features2 = model.encode_text(text2)
    # compute cosine similarity
    similarity = text_features @ text_features2.T
    # normalize similarity
    similarity = similarity / (
        text_features.norm(dim=-1, keepdim=True)
        * text_features2.norm(dim=-1, keepdim=True).T
    )
    return similarity.item()


"""
3. Refine
"""


"""
Others
"""


def get_parser():
    parser = argparse.ArgumentParser(description="Configuration Beyond Fixed Forms")
    parser.add_argument("--config", type=str, required=True, help="Config")
    parser.add_argument("--cls", type=str, required=True, help="Class")
    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()
    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_prompt = args.cls
    scene_checkpoint = read_scene_checkpoint(text_prompt)
    
    clip_model, _ = clip.load("ViT-L/14", device=device)

    stage1_dir = cfg.stage_1_results_dir
    mask_3d_dir = cfg.mask_3d_dir

    stage2_output_dir = os.path.join(mask_3d_dir, text_prompt)
    
    stage2_outputs = sorted([s for s in os.listdir(stage2_output_dir) if s.endswith("_00.pth")])
    # stage2_outputs = ["scene0435_00.pth"]
    
    
    """"First Iteration for Thresholds"""
    all_ious = []
    all_similarities = [] # shape (num_scene, num_masks_in_scene)
    all_matched_stage1_masks = [] # shape (num_scene, num_masks_in_scene)
    all_matched_stage2_masks = [] # shape (num_scene, num_masks_in_scene)
    all_stage2_conf = [] # shape (num_scene, num_masks_in_scene)
    
    for stage2_output in tqdm(stage2_outputs, desc="Select thresholds for refinement"):
        scene_id = stage2_output.replace(".pth", "")
        # if scene_checkpoint.get(scene_id, False):
        #     continue
        # print("Working on", scene_id)

        stage1_path = os.path.join(stage1_dir, f"{scene_id}.pth")
        stage2_path = os.path.join(mask_3d_dir, text_prompt, f"{scene_id}.pth")

        if os.path.exists(stage1_path) and os.path.exists(stage2_path):
            pass
        else:
            continue
        
        # print(stage1_path, stage2_path)

        stage1_output = torch.load(stage1_path, map_location="cpu")
        stage2_output = torch.load(stage2_path, map_location="cpu")	
        
        """If stage 2 is empty, save empty output and continue"""
        if len(stage2_output["conf"]) == 0:
            print("Empty stage 2 mask")
            all_ious.append([])
            all_similarities.append([])
            all_matched_stage1_masks.append([])
            all_matched_stage2_masks.append([])
            all_stage2_conf.append([])
            continue

        # Process stage 1 masks
        instance = stage1_output["ins"]
        stage1_output["ins"] = torch.stack(
            [torch.tensor(rle_decode(ins)) for ins in instance]
        )

        # Process stage 1 labels
        class_indices = stage1_output["final_class"]
        stage1_output["final_class"] = [idx_to_label(idx) for idx in class_indices]

        # compute iou between stage1 and stage2 masks
        iou_matrix = calculate_iou_between_stages(
            stage1_output["ins"], stage2_output["ins"]
        )
        max_match = torch.argmax(
            iou_matrix, dim=1
        )  # for each stage2 mask, find the best matched stage1 mask

        # compute similarity between stage1 and stage2 labels
        matched_labels = [stage1_output["final_class"][idx] for idx in max_match]
        text2 = text_prompt
        clip_similarities = [
            float(compute_clip_similarity(clip_model, text2, text1))
            for text1 in matched_labels
        ]
        
        print("iou for matched labels", iou_matrix[range(len(max_match)), max_match])
        print("Matched labels:", matched_labels)
        print("Clip similarities:", clip_similarities)
        
        all_ious.append(iou_matrix[range(len(max_match)), max_match])
        all_similarities.append(clip_similarities)
        all_matched_stage1_masks.append(stage1_output["ins"][max_match])
        all_matched_stage2_masks.append(stage2_output["ins"])
        all_stage2_conf.append(stage2_output["conf"])  
    
    
    """"determine thresholds"""
    iou_thres = cfg.refiment_iou_thres
    
    # sim_thres = cfg.refinment_sim_thres
    sim_percentile = cfg.refinment_sim_percentile
    flattern_sim = [sim for sims in all_similarities for sim in sims]
    sim_unique = sorted(set(flattern_sim))
    print("Unique similarities:", sim_unique)
    sim_thres = sim_unique[int(len(sim_unique) * sim_percentile)]
    print("Final thresholds:", iou_thres, sim_thres)
    
    
    
    
    for s, stage2_output in tqdm(enumerate(stage2_outputs), desc="Refining stage1 output with stage2 outcomes"):
        scene_id = stage2_output.replace(".pth", "")
        
        # use stage2 masks to refine stage1 masks
        final_output = {
            "ins": [],  # (Ins, N) torch.Tensor
            "conf": [],  # (Ins, ) torch.Tensor
            "final_class": [],  # (Ins,) List[str]
        }
        
        ious = all_ious[s]
        if len(ious) == 0:
            # print("Empty stage 2 mask")
            os.makedirs(os.path.join(cfg.final_output_dir, text_prompt), exist_ok=True)
            torch.save(
                stage2_output,
                os.path.join(cfg.final_output_dir, text_prompt, f"{scene_id}.pth"),
            )
        
        for m, iou in enumerate(ious) :  # i in stage2, idx in stage1
            # i in stage2, idx in stage1
            if iou > iou_thres:
                # use stage1 mask
                if all_similarities[s][m] < sim_thres:
                    continue

                final_output["ins"].append(all_matched_stage1_masks[s][m])
                # use intersection of stage1 and stage2 mask
                # final_output["ins"].append(
                #     all_matched_stage1_masks[s][m] & all_matched_stage2_masks[s][m]
                # )
                # final_output["conf"].append(stage1_output["conf"][idx])
                # use corresponding stage2 conf and label
                final_output["conf"].append(all_stage2_conf[s][m])
                final_output["final_class"].append(text_prompt)
            else:
                # # use intersection of stage1 and stage2 mask
                final_output["ins"].append(
                    all_matched_stage1_masks[s][m] & all_matched_stage2_masks[s][m]
                )
                # use stage2 mask
                # final_output["ins"].append(all_matched_stage2_masks[s][m])
                # # use corresponding stage2 conf
                final_output["conf"].append(all_stage2_conf[s][m])
                # average confidence
                # final_output["conf"].append(
                #     (stage1_output["conf"][idx] + 4 * stage2_output["conf"][i]) / 2
                # )
                # use corresponding stage2 label
                final_output["final_class"].append(text_prompt)

        # transform to torch.Tensor
        
        if len(final_output["ins"]) == 0:
            # print("No mask after refinement")
            os.makedirs(os.path.join(cfg.final_output_dir, text_prompt), exist_ok=True)
            torch.save(
                final_output,
                os.path.join(cfg.final_output_dir, text_prompt, f"{scene_id}.pth"),
            )
            continue
            
        final_output["ins"] = torch.stack(final_output["ins"]).to(bool)
        final_output["conf"] = torch.stack(final_output["conf"])
        # print(
        #     "Refined output shape:",
        #     final_output["ins"].shape,
        #     final_output["conf"].shape,
        #     len(final_output["final_class"]),
        # )
        # print("Refined output classes:", final_output["final_class"])

        # save the refined masks
        os.makedirs(os.path.join(cfg.final_output_dir, text_prompt), exist_ok=True)
        torch.save(
            final_output,
            os.path.join(cfg.final_output_dir, text_prompt, f"{scene_id}.pth"),
        )
        scene_checkpoint[scene_id] = True
        write_scene_checkpoint(text_prompt, scene_checkpoint)
