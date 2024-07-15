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

    """perform non-maximum suppression on the output pointcloud masks

    args:
        output: Dict, keys=["ins", "conf", "final_class"]
        nms_thres: float, threshold for nms

    returns:
        Dict, keys=["ins", "conf", "final_class"]
    """
    ins = output["ins"] # (Ins, N) torch.Tensor
    conf = output["conf"]
    final_class = output["final_class"]
    if len(ins) == 0:
        return output
    # sort by confidence
    sorted_idx = torch.argsort(conf, descending=True)
    ins = ins[sorted_idx]
    conf = conf[sorted_idx]
    final_class = [final_class[i] for i in sorted_idx]
    # perform nms
    keep = []
    while len(ins) > 0:
        keep.append(ins[0])
        iou = (ins[0] & ins[1:]).sum(dim=-1) / (ins[0] | ins[1:]).sum(dim=-1)
        ins = ins[1:][iou < nms_thres]
    return {
        "ins": torch.stack(keep),
        "conf": torch.tensor([1.0] * len(keep)),
        "final_class": final_class,
    }







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
    output_ids = [s.replace(".pth", "") for s in stage2_outputs]
    for output_id in output_ids:
        if output_id not in ['scene0011_00', 'scene0015_00', 'scene0019_00', 'scene0025_00', 'scene0030_00', 'scene0046_00', 'scene0050_00', 'scene0063_00', 'scene0064_00', 'scene0077_00', 'scene0081_00', 'scene0084_00', 'scene0086_00', 'scene0088_00', 'scene0095_00', 'scene0100_00', 'scene0131_00', 'scene0139_00', 'scene0146_00', 'scene0149_00', 'scene0153_00', 'scene0164_00', 'scene0169_00', 'scene0187_00', 'scene0193_00', 'scene0196_00', 'scene0203_00', 'scene0207_00', 'scene0208_00', 'scene0217_00', 'scene0221_00', 'scene0222_00', 'scene0231_00', 'scene0246_00', 'scene0249_00', 'scene0251_00', 'scene0256_00', 'scene0257_00', 'scene0277_00', 'scene0278_00', 'scene0300_00', 'scene0304_00', 'scene0307_00', 'scene0314_00', 'scene0316_00', 'scene0328_00', 'scene0329_00', 'scene0334_00', 'scene0338_00', 'scene0342_00', 'scene0343_00', 'scene0351_00', 'scene0353_00', 'scene0354_00', 'scene0355_00', 'scene0356_00', 'scene0357_00', 'scene0377_00', 'scene0378_00', 'scene0382_00', 'scene0389_00', 'scene0406_00', 'scene0412_00', 'scene0414_00', 'scene0423_00', 'scene0426_00', 'scene0427_00', 'scene0430_00', 'scene0432_00', 'scene0435_00', 'scene0441_00', 'scene0458_00', 'scene0461_00', 'scene0462_00', 'scene0474_00', 'scene0488_00', 'scene0490_00', 'scene0494_00', 'scene0496_00', 'scene0500_00', 'scene0518_00', 'scene0527_00', 'scene0535_00', 'scene0549_00', 'scene0550_00', 'scene0552_00', 'scene0553_00', 'scene0558_00', 'scene0559_00', 'scene0565_00', 'scene0568_00', 'scene0574_00', 'scene0575_00', 'scene0578_00', 'scene0580_00', 'scene0583_00', 'scene0591_00', 'scene0593_00', 'scene0595_00', 'scene0598_00', 'scene0599_00', 'scene0606_00', 'scene0607_00', 'scene0608_00', 'scene0609_00', 'scene0616_00', 'scene0618_00', 'scene0621_00', 'scene0629_00', 'scene0633_00', 'scene0643_00', 'scene0644_00', 'scene0645_00', 'scene0647_00', 'scene0648_00', 'scene0651_00', 'scene0652_00', 'scene0653_00', 'scene0655_00', 'scene0658_00', 'scene0660_00', 'scene0663_00', 'scene0664_00', 'scene0665_00', 'scene0670_00', 'scene0671_00', 'scene0678_00', 'scene0684_00', 'scene0685_00', 'scene0686_00', 'scene0689_00', 'scene0690_00', 'scene0693_00', 'scene0695_00', 'scene0696_00', 'scene0697_00', 'scene0699_00', 'scene0700_00', 'scene0701_00', 'scene0702_00', 'scene0704_00']:
            print(f'DEBUG------------------------------------{output_id}--------------------------------')
    # stage2_outputs = ["scene0435_00.pth"]
    
    
    """"First Iteration for Thresholds"""
    all_ious = []
    # all_max_match = []
    all_similarities = [] # shape (num_scene, num_masks_in_scene)
    all_matched_stage1_masks = [] # shape (num_scene, num_masks_in_scene)
    all_matched_stage2_masks = [] # shape (num_scene, num_masks_in_scene)
    all_stage2_conf = [] # shape (num_scene, num_masks_in_scene)
    
    for stage2_output in tqdm(stage2_outputs, desc="Select thresholds for refinement"):
        scene_id = stage2_output.replace(".pth", "")
        # if scene_checkpoint.get(scene_id, False):
        #     continue
        print("Working on", scene_id)

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
            # all_max_match.append([])
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

        # Perform NMS on stage 1 masks
        # stage1_output = nms(stage1_output, cfg.nms_thres)

        # Process stage 1 labels
        class_indices = stage1_output["final_class"]
        stage1_output["final_class"] = [idx_to_label(idx) for idx in class_indices]

        # compute iou between stage1 and stage2 masks
        iou_matrix = calculate_iou_between_stages(
            stage1_output["ins"], stage2_output["ins"]
        )
        max_match = torch.argmax(
            iou_matrix, dim=1
        )  # for each stage2 mask, find the best matched stage1 mask. shape=(m,)



        """ merge stage2 masks with the same stage1 mask matched"""
        # uniques, counts = torch.unique(max_match, return_counts=True)
        matched_stage1_iou_matrix = calculate_iou_between_stages(
            stage1_output["ins"][max_match], stage1_output["ins"][max_match]
        ) # shape=(m, m), iou between matched stage1 masks
        # set diagonal to 0
        matched_stage1_iou_matrix[range(len(max_match)), range(len(max_match))] = 0
        # print("DEBUG matched_stage1_iou_matrix", matched_stage1_iou_matrix)
        # find matched stage1 masks with iou > cfg.stage1_iou_thres
        matched_stage1_iou_matrix = (matched_stage1_iou_matrix > cfg.stage1_iou_thres).to(int)
        print("DEBUG matched_stage1_iou_matrix", matched_stage1_iou_matrix)
        # merge stage1 masks with iou > cfg.stage1_iou_thres


        
        best_match_after_iou_check = []
        remove_idx = torch.ones(len(matched_stage1_iou_matrix), dtype=torch.int) * -1
        for i in range(len(matched_stage1_iou_matrix)):
            print("DEBUG i", i)
            if remove_idx[i] != -1:
                idx = remove_idx[i]
                print("DEBUG idx", idx)
                best_match_after_iou_check.append(max_match[idx]) # add the same stage1 mask
                continue

            best_match_after_iou_check.append(max_match[i])

            if matched_stage1_iou_matrix[i].sum() > 0:
               for j in range(len(matched_stage1_iou_matrix[i])):
                #    print(matched_stage1_iou_matrix[i])
                   if matched_stage1_iou_matrix[i][j] == 1:
                    #    print("DEbug ih", i, j)
                       remove_idx[j] = i
                       stage1_output["ins"][max_match[i]] = stage1_output["ins"][max_match[i]] | stage1_output["ins"][max_match[j]]
                       print("DEBUG remove_idx", remove_idx)

        print("DEBUG best_match_after_iou_check", best_match_after_iou_check)
            # match = torch.nonzero(matched_stage1_iou_matrix[i]).squeeze(1)
            # if len(match) > 0:
            #     best_match_after_iou_check.append(match[0])
            #     remove_idx.extend(match[1:])
            # else:
            #     best_match_after_iou_check.append(i)
        best_match_after_iou_check = torch.tensor(best_match_after_iou_check)
        uniques, counts = torch.unique(best_match_after_iou_check, return_counts=True)

        print("DEBUG uniques", uniques, counts)
        for i, count in zip(uniques, counts):
            print(f"Matched {count} times with {i}")
            if count > 1:
                print("Merge stage2 masks")
                masks = stage2_output["ins"][best_match_after_iou_check == i]
                print("DEBUG masks", masks.shape, i, [best_match_after_iou_check == i])
                merged_mask = masks.any(dim=0)
                confs = stage2_output["conf"][best_match_after_iou_check == i]
                merged_conf = confs.mean()
                # remove other merged masks
                stage2_output["ins"] = torch.cat(
                    [stage2_output["ins"][best_match_after_iou_check != i], merged_mask.unsqueeze(0)]
                ) # shape=(m, x)
                stage2_output["conf"] = torch.cat(
                    [stage2_output["conf"][best_match_after_iou_check != i], merged_conf.unsqueeze(0)]
                )
                best_match_after_iou_check = torch.cat(
                    [best_match_after_iou_check[best_match_after_iou_check != i], i.unsqueeze(0)]
                )


        # compute iou between stage1 and stage2 masks
        iou_matrix = calculate_iou_between_stages(
            stage1_output["ins"], stage2_output["ins"]
        )
        max_match = torch.argmax(
            iou_matrix, dim=1
        )  # for each stage2 mask, find the best matched stage1 mask. shape=(m,)
                
        

       


        """compute similarity between stage1 and stage2 labels"""
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
        # all_max_match.append(max_match)
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
        print("DEBUG all ious", all_ious[s])
        
        ious = all_ious[s]
        if len(ious) == 0:
            # print("Empty stage 2 mask")
            os.makedirs(os.path.join(cfg.final_output_dir, text_prompt), exist_ok=True)
            torch.save(
                final_output,
                os.path.join(cfg.final_output_dir, text_prompt, f"{scene_id}.pth"),
            )
            continue
        
        for m, iou in enumerate(ious) :  # i in stage2, idx in stage1
            # i in stage2, idx in stage1
            if iou > iou_thres:
                print("USE STAGE 1 MASK", iou)
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
                print("USE STAGE 2 MASK", iou)
                # # use intersection of stage1 and stage2 mask
                # final_output["ins"].append(
                #     all_matched_stage1_masks[s][m] & all_matched_stage2_masks[s][m]
                # )
                # use stage2 mask
                final_output["ins"].append(all_matched_stage2_masks[s][m])
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
        # scene_checkpoint[scene_id] = True
        # write_scene_checkpoint(text_prompt, scene_checkpoint)
