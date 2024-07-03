##########################################
# Navigate to Beyond-Fixed-Forms directory and run:
# python evaluation/eval/run_evl.py --cls "clothes"
##########################################
import os
import argparse
import numpy as np
import torch

import sys
sys.path.append("./")

from evaluation.dataset.scannet200 import INSTANCE_CAT_SCANNET_200, BENCHMARK_SEMANTIC_IDXS, HEAD_CATS_SCANNET_200,COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200, BASE_CLASSES_SCANNET200, NOVEL_CLASSES_SCANNET200
from evaluation.eval.scannetv2_inst_eval import ScanNetEval
from tqdm import tqdm

def rle_decode(rle):
    length = rle["length"]
    s = rle["counts"]

    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask

def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate ScanNet200")
    parser.add_argument("--cls", type=str, required=True, help="Specific class to evaluate")
    return parser

def read_existing_results(filepath):
    head_common_tail = True
    base_novel = False
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            return file.readlines()
    else:
        # Create a header if the file does not exist
        header = ["class,class id,ap,ap50,ap25,rc,rc50,rc25\n"]
        if head_common_tail:
            return header + ["\n Head Classes: \n"] + [f"{cls},-,-,-\n" for cls in HEAD_CATS_SCANNET_200] + ["\n Common Classes: \n"] + [f"{cls},-,-,-\n" for cls in COMMON_CATS_SCANNET_200] + [" \n Tail Classes: \n"] + [f"{cls},-,-,-\n" for cls in TAIL_CATS_SCANNET_200]
        elif base_novel:
            return header + ["\n Base Classes:\n"] + [f"{cls},-,-,-\n" for cls in BASE_CLASSES_SCANNET200] + ["\n Novel Classes:\n"] + [f"{cls},-,-,-\n" for cls in NOVEL_CLASSES_SCANNET200]
        else:
            return header + [f"{cls},-,-,-\n" for cls in INSTANCE_CAT_SCANNET_200]

def write_results(filepath, results):
    print("DEBUG writing results to", filepath)
    with open(filepath, 'w') as file:
        file.writelines(results)

def update_results(existing_results, new_results, class_name):
    updated_results = []
    for line in existing_results:
        if line.startswith(f"{class_name},"):
            updated_results.append(new_results)
        else:
            updated_results.append(line)
    return updated_results

scan_eval = ScanNetEval(class_labels=INSTANCE_CAT_SCANNET_200)
# data_path = "../exp/version_qualitative/final_result_hier_agglo_2d"
# data_path = "./exp_stage_1/Result_OpenVocab_ISBNet-GSAM/final_result_hier_agglo"



if __name__ == "__main__":
    args = get_parser().parse_args()
    class_to_evaluate = args.cls

    data_path = f"./output/final_output/{class_to_evaluate}"
    pcl_path = "./data/Scannet200/Scannet200_3D/groundtruth"
    results_filepath = "./evaluation/eval_results/overall_results.txt"
    scenes = sorted([s for s in os.listdir(data_path) if s.endswith(".pth")])

    gtsem = []
    gtinst = []
    res = []

    for scene in tqdm(scenes, desc="Evaluating class "+class_to_evaluate+"..."):        
        # print("Working on",scene)
        gt_path = os.path.join(pcl_path, scene)
        loader = torch.load(gt_path, map_location="cpu")

        sem_gt, inst_gt = loader[2], loader[3]
        ## !! the sem_gt here has index > 200, need to convert to 0-199
        ## assign_instances_for_scan will do gt_sem = gt_sem - 2 + 1
        # print("DEBUG all unique semantic labels and their occurance in scene",scene, np.unique(sem_gt.astype(np.int32), return_counts=True))
        sem_gt = [BENCHMARK_SEMANTIC_IDXS.index(int(s)) if s!=0 and int(s) in BENCHMARK_SEMANTIC_IDXS else -1 for s in sem_gt]
        # print("debug idx for gt index (32)", BENCHMARK_SEMANTIC_IDXS.index(32))
        # print("label index for clothes(25)", BENCHMARK_SEMANTIC_IDXS[25])
        # print("DEBUG GT find number of 25 in gt", np.sum(np.array(sem_gt) == 25))
        
        gtsem.append(np.array(sem_gt).astype(np.int32))
        gtinst.append(np.array(inst_gt).astype(np.int32))
        # print("DEBUG GT", len(sem_gt), len(inst_gt))
              
        scene_path = os.path.join(data_path, scene)
        pred_mask = torch.load(scene_path, map_location="cpu")

        masks, category, score = pred_mask["ins"], pred_mask["final_class"], pred_mask["conf"]

        masks = torch.tensor(masks)
        score = torch.tensor(score)
        # print("DEBUG", masks.shape, category, score)
        # if category is not tensor
        if not torch.is_tensor(category):
            # print("DEBUG This is BeyondFF output, converting to tensor")
            # then it is a list of str class names, convert to tensor
            category = torch.tensor([INSTANCE_CAT_SCANNET_200.index(c.lower()) for c in category])
            # print("DEBUG", category, masks.shape, score.shape)
            # print("DEBUG find number of 25 in pred", np.sum(np.array(masks) == 1))
            
        
        n_mask = category.shape[0]
        tmp = []
        for ind in range(n_mask):
            if isinstance(masks[ind], dict):
                mask = rle_decode(masks[ind])
            else:
                mask = (masks[ind] == 1).numpy().astype(np.uint8)

            # conf = score[ind] #

            conf = 1.0
            final_class = float(category[ind])
            scene_id = scene.replace(".pth", "")
            tmp.append({"scan_id": scene_id, "label_id": final_class + 1, "conf": conf, "pred_mask": mask})

        res.append(tmp)

    avgs = scan_eval.evaluate(res, gtsem, gtinst)

    # Read existing results
    existing_results = read_existing_results(results_filepath)
    # Prepare new results for the current class
    print("DEBUG avg value",avgs["classes"][class_to_evaluate])
    new_results = ",".join([class_to_evaluate, str(avgs["classes"][class_to_evaluate]["ap"]) , str(avgs["classes"][class_to_evaluate]["ap50%"]) , str(avgs["classes"][class_to_evaluate]["ap25%"]),  str(avgs["classes"][class_to_evaluate]["rc"]),  str(avgs["classes"][class_to_evaluate]["rc50%"]),  str(avgs["classes"][class_to_evaluate]["rc25%"]),"\n"])
    #f"{avgs["classes"][class_to_evaluate]["ap"]},{avgs["classes"][class_to_evaluate]["ap50%"]},{avgs["classes"][class_to_evaluate]["ap25%"]}\n"
    # Update the results
    updated_results = update_results(existing_results, new_results, class_to_evaluate)
    # Write the updated results back to the file
    write_results(results_filepath, updated_results)
