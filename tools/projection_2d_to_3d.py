import numpy as np

# import open3d as o3d
import cv2
import yaml

import matplotlib.pyplot as plt
import copy
import torch
import os
import argparse

from configs import config as cfg
from munch import Munch

from tqdm import tqdm
from typing import List, Dict, Tuple 
import sys

sys.path.append("/medar_smart/temp/Beyond-Fixed-Forms/tools")
# from segmentation_2d import inference_grounded_sam
from utils.rle_encode_decode import encode_2d_masks, decode_2d_masks

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


"""
1. Project 2d masks to 3d point cloud
"""


def compute_projected_pts_tensor(pts, cam_intr):
    # map 3d pointclouds in camera coordinates system to 2d
    # pts shape (N, 3)

    pts = pts.T  # (3, N)
    # print("cam_int", cam_intr)
    projected_pts = cam_intr @ pts / pts[2]  # (3, N)
    # print("pts0", pts[:,0])
    # print("projected_pts0", (cam_intr @ pts[:,0]).astype(np.int64))
    projected_pts = projected_pts[:2].T  # (N, 2)
    projected_pts = (np.round(projected_pts)).astype(np.int64)
    return projected_pts


def compute_visibility_mask_tensor(pts, projected_pts, depth_im, depth_thresh=0.005):
    # compare z in camera coordinates and depth image
    # to check if there projected points are visible
    im_h, im_w = depth_im.shape
    # print("depth_im shape", depth_im.shape)
    visibility_mask = np.zeros(projected_pts.shape[0]).astype(np.bool_)
    inbounds = (
        (projected_pts[:, 0] >= 0)
        & (projected_pts[:, 0] < im_w)
        & (projected_pts[:, 1] >= 0)
        & (projected_pts[:, 1] < im_h)
    )  # (N,)
    projected_pts = projected_pts[inbounds]  # (X, 2)
    depth_check = (depth_im[projected_pts[:, 1], projected_pts[:, 0]] != 0) & (
        np.abs(pts[inbounds][:, 2] - depth_im[projected_pts[:, 1], projected_pts[:, 0]])
        < depth_thresh
    )

    visibility_mask[inbounds] = depth_check
    return visibility_mask  # (N,)


def compute_visible_masked_pts_tensor(
    scene_pts, projected_pts, visibility_mask, pred_masks
):
    # return masked 3d points
    N = scene_pts.shape[0]
    M, H, W = pred_masks.shape  # (M, H, W)
    # print("DEBUG mask shape value", H, W)
    # print("DEBUG M value", M)
    masked_pts = np.zeros((M, N), dtype=np.bool_)
    visiable_pts = projected_pts[visibility_mask]  # (X, 2)
    for m in range(M):
        x, y = visiable_pts.T  # (X,)
        # # Ensure x and y are within bounds
        # valid_indices = (x >= 0) & (x < pred_masks.shape[2]) & (y >= 0) & (y < pred_masks.shape[1])
        # x, y = x[valid_indices], y[valid_indices]

        mask_check = pred_masks[m, y, x]  # (X,)
        masked_pts[m, visibility_mask] = mask_check

    return masked_pts


"""
2. Aggregating 3d masks
"""


def aggregate(
    backprojected_3d_masks: dict, iou_threshold=0.25, feature_similarity_threshold=0.75
) -> dict:
    """
    calculate iou
    calculate feature similarity

    if iou >= threshold and feature similarity >= threshold:
        aggregate
    else:
        create new mask
    """
    labels = backprojected_3d_masks["final_class"]  # List[str]
    semantic_matrix = calculate_feature_similarity(labels)

    ins_masks = backprojected_3d_masks["ins"].to(device)  # (Ins, N)
    iou_matrix = calculate_iou(ins_masks)

    confidences = backprojected_3d_masks["conf"].to(device)  # (Ins, )

    merge_matrix = semantic_matrix & (
        iou_matrix > iou_threshold
    )  # dtype: bool (Ins, Ins)

    # aggregate masks with high iou
    (
        aggregated_masks,
        aggregated_confidences,
        aggregated_labels,
        mask_indeces_to_be_merged,
    ) = merge_masks(ins_masks, confidences, labels, merge_matrix)

    if mask_indeces_to_be_merged == []:
        return {
            "ins": torch.tensor([[]]).to(device=device),  # (Ins, N)
            "conf": torch.tensor([]).to(device=device),  # (Ins, )
            "final_class": [],  # (Ins,)
        }
    
    # solve overlapping
    final_masks = solve_overlapping(aggregated_masks, mask_indeces_to_be_merged)

    return {
        "ins": final_masks,  # torch.tensor (Ins, N)
        "conf": aggregated_confidences,  # torch.tensor (Ins, )
        "final_class": aggregated_labels,  # List[str] (Ins,)
    }


def calculate_iou(ins_masks: torch.Tensor) -> torch.Tensor:
    """calculate iou between all masks

    args:
        ins_masks: torch.tensor (Ins, N)

    return:
        iou_matrix: torch.tensor (Ins, Ins)
    """
    ins_masks = ins_masks.float()
    intersection = torch.matmul(ins_masks, ins_masks.T)  # (Ins, Ins)
    union = (
        torch.sum(ins_masks, dim=1).unsqueeze(1)
        + torch.sum(ins_masks, dim=1).unsqueeze(0)
        - intersection
    )
    iou_matrix = intersection / union
    return iou_matrix


def calculate_feature_similarity(labels: List[str]) -> torch.Tensor:
    """calculate feature similarity between all masks

    args:
        labels: list[str]

    return:
        feature_similarity_matrix: torch.tensor (Ins, Ins)
    """  # TODO: add clip feature similarity
    feature_similarity_matrix = torch.zeros(len(labels), len(labels), device=device)
    for i in range(len(labels)):
        for j in range(i, len(labels)):
            if labels[i] == labels[j]:
                feature_similarity_matrix[i, j] = 1
                feature_similarity_matrix[j, i] = 1

    # convert to boolean
    feature_similarity_matrix = feature_similarity_matrix.bool()
    return feature_similarity_matrix


def merge_masks(
    ins_masks: torch.Tensor,
    confidences: torch.Tensor,
    labels: List[str],
    merge_matrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[List[int]]]:

    # find masks to be merged
    merge_matrix = merge_matrix.float()
    mask_indeces_to_be_merged = find_unconnected_subgraphs_tensor(merge_matrix)
    
    # filter mask aggregated from less than 3 masks

    mask_indeces_to_be_merged = [ mask_indeces for mask_indeces in mask_indeces_to_be_merged if len(mask_indeces) >= cfg.min_aggragated_masks]

    

    print("masks_to_be_merged", mask_indeces_to_be_merged)


    # merge masks
    aggregated_masks = []
    aggregated_confidences = []
    aggregated_labels = []
    for mask_indeces in mask_indeces_to_be_merged:

        if mask_indeces == []:
            continue

        mask = torch.zeros(ins_masks.shape[1], dtype=torch.bool, device=device)
        conf = []
        for index in mask_indeces:
            mask |= ins_masks[index]
            conf.append(confidences[index])
        aggregated_masks.append(mask)
        aggregated_confidences.append(sum(conf) / len(conf))
        aggregated_labels.append(labels[mask_indeces[0]])



    if len(aggregated_masks) == 0:
        return (
            torch.tensor([[]]).to(device=device),  # (Ins, N)
            torch.tensor([]).to(device=device),  # (Ins, )
            [],  # (Ins,)
            [],
        )
    
    # convert type 
    aggregated_masks = torch.stack(aggregated_masks)  # (Ins, N)
    aggregated_confidences = torch.tensor(aggregated_confidences)  # (Ins, )

    return (
        aggregated_masks,
        aggregated_confidences,
        aggregated_labels,
        mask_indeces_to_be_merged,
    )


def find_unconnected_subgraphs_tensor(adj_matrix: torch.Tensor) -> List[List[int]]:
    num_nodes = adj_matrix.size(0)
    # Create an identity matrix for comparison
    identity = torch.eye(num_nodes, dtype=torch.float32)
    # Start with the adjacency matrix itself
    reachability_matrix = adj_matrix.clone()

    # Repeat matrix multiplication to propagate connectivity
    for _ in range(num_nodes):
        reachability_matrix = torch.matmul(reachability_matrix, adj_matrix) + adj_matrix
        reachability_matrix = torch.clamp(reachability_matrix, 0, 1)

    # Identify unique connected components
    components = []
    visited = torch.zeros(num_nodes, dtype=torch.bool)
    for i in range(num_nodes):
        if not visited[i]:
            component_mask = reachability_matrix[i] > 0
            component = torch.nonzero(component_mask, as_tuple=False).squeeze().tolist()
            # Ensure component is a list even if it's a single element
            component = [component] if isinstance(component, int) else component
            components.append(component)
            visited[component_mask] = True

    return components


def solve_overlapping(
    aggregated_masks: torch.Tensor,
    mask_indeces_to_be_merged: List[List[int]],
) -> torch.Tensor:
    """
    solve overlapping among all masks
    """
    # number of aggrated inital masks in each aggregated mask
    num_masks = [len(mask_indeces) for mask_indeces in mask_indeces_to_be_merged]

    # find overlapping masks in aggregated_masks
    overlapping_masks = []
    for i in range(len(aggregated_masks)):
        for j in range(i + 1, len(aggregated_masks)):
            if torch.any(aggregated_masks[i] & aggregated_masks[j]):
                overlapping_masks.append((i, j))

    # only keep overlapped points for masks aggregated from more masks
    for i, j in overlapping_masks:
        if num_masks[i] > num_masks[j]:
            aggregated_masks[j] &= ~aggregated_masks[i]
        else:
            aggregated_masks[i] &= ~aggregated_masks[j]

    return aggregated_masks


"""
3. Filtering 3d masks
"""  # TODO: rewrite filtering in __main__ to a function here


"""
*: Help functions
"""


def get_parser():
    parser = argparse.ArgumentParser(description="Configuration Open3DIS")
    parser.add_argument("--config", type=str, required=True, help="Config")
    parser.add_argument("--cls", type=str, required=True, help="Class")
    return parser

def scene_checkpoint_file(class_name):
    # return f"projection_2d_to_3d_checkpoint_{class_name}.yaml"
    return f"checkpoints/projection_2d_to_3d_checkpoint.yaml"

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

if __name__ == "__main__":

    args = get_parser().parse_args()
    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    depth_scale = 1000  # not sure, JUST TRYING!

    # load camera configs
    # cam_intr = np.loadtxt("/home/jie_zhenghao/Open3DIS/data/Scannet200/Scannet200_2D_5interval/val/scene0435_00/intrinsic.txt")
    # cam_intr =np.array(
    #             [[577.870605, 0.0, 319.5],
    #              [0.0, 577.870605, 239.5],
    #              [0.0, 0.0, 1.0]])
    
    
    mask_2d_dir = cfg.mask_2d_dir
    scene_2d_dir = cfg.scene_2d_dir
    
    text_prompt = args.cls
    scene_checkpoint = read_scene_checkpoint(text_prompt)
    seg_output_dir = os.path.join(mask_2d_dir, text_prompt)
    
    seg_outputs = sorted([s for s in os.listdir(seg_output_dir) if s.endswith("_00.pth")])
    # seg_outputs = ["scene0353_00.pth"]
    for seg_output in tqdm(seg_outputs, desc="Projecting 2d masks to 3d point cloud"):
        scene_id = seg_output[:-4]
        print("Working on", scene_id)
        if scene_checkpoint.get(scene_id, False):
            continue
        cam_intr_path = os.path.join(scene_2d_dir, scene_id, "intrinsic", "intrinsic_color.txt")

        # intrinsic color
        # cam_intr = np.array(
        #     [[1170.1, 0.0, 647.7], [0.0, 1170.187988, 483.750000], [0.0, 0.0, 1.0]]
        # )
        cam_intr = np.loadtxt(cam_intr_path)[:3,:3]
        cam_pose_dir = os.path.join(scene_2d_dir, scene_id, "pose")
        depth_im_dir = os.path.join(scene_2d_dir, scene_id, "depth")
        
        # print(cam_intr)
        # print(cam_intr.shape)
        # cam_intr = np.asarray(cam_config['cam_intr'])

        # load 3d point cloud
        scene_pcd_path = os.path.join(cfg.scene_npy_dir, f"{scene_id}.npy")
        # scene_pcd = o3d.io.read_point_cloud(str(scene_pcd_path))
        scene_pcd = np.load(scene_pcd_path)[:, :3]
        scene_pcd = np.concatenate(
            [scene_pcd, torch.ones([scene_pcd.shape[0], 1])], axis=1
        ).T
        # print("scene_pcd shape:", scene_pcd.shape)

        # 2d sam masks
        # annotated_frame, segmented_frame_masks = inference_grounded_sam()
        masks_2d_path = os.path.join(mask_2d_dir, text_prompt, f"{scene_id}.pth")
        gronded_sam_results = torch.load(masks_2d_path)
        
        # convert rle to masks
        # print("converting rle to masks")
        gronded_sam_results = decode_2d_masks(gronded_sam_results, (cfg.height_2d, cfg.width_2d))

        masked_counts = torch.zeros(scene_pcd.shape[1]).to(
            device=device
        )  # scene_pcd has shape (4, N)

        # 3d mask aggregation
        backprojected_3d_masks = {
            "ins": [],  # (Ins, N)
            "conf": [],  # (Ins, )
            "final_class": [],  # (Ins,)
        }

        for i in range(len(gronded_sam_results)):  

            frame_id = gronded_sam_results[i]["frame_id"][:-4]
            # print("-------------------------frame", frame_id, "-------------------------")
            segmented_frame_masks = gronded_sam_results[i]["segmented_frame_masks"].to(torch.float32)  # (M, 1, W, H)
            confidences = gronded_sam_results[i]["confidences"]
            labels = gronded_sam_results[i]["labels"]

            pred_masks = segmented_frame_masks.squeeze(dim=1).numpy()  # (M, H, W)
            cam_pose = np.loadtxt(os.path.join(cam_pose_dir, f"{frame_id}.txt"))

            scene_pts = copy.deepcopy(scene_pcd)
            scene_pts = (np.linalg.inv(cam_pose) @ scene_pts).T[:, :3]  # (N, 3)
            projected_pts = compute_projected_pts_tensor(scene_pts, cam_intr)

            photo_width = int(cfg.width_2d)
            photo_height = int(cfg.height_2d)
            # frame_id_num = frame_id.split('.')[0]
            depth_im_path = os.path.join(depth_im_dir, f"{frame_id}.png")
            depth_im = (
                cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                / depth_scale
            )
            depth_im = cv2.resize(depth_im, (photo_width, photo_height))  # Width x Height
            visibility_mask = compute_visibility_mask_tensor(
                scene_pts, projected_pts, depth_im, depth_thresh=0.08
            )

            masked_pts = compute_visible_masked_pts_tensor(
                scene_pts, projected_pts, visibility_mask, pred_masks
            )

            masked_pts = torch.from_numpy(masked_pts).to(device)  # (M, N)
            mask_area = torch.sum(masked_pts, dim=1).detach().cpu().numpy()  # (M,)
            # print(
            #     "number of 3d mask points:",
            #     mask_area,
            #     "number of 2d masks:",
            #     pred_masks.sum(axis=(1, 2)),
            # )

            for i in range(masked_pts.shape[0]):
                backprojected_3d_masks["ins"].append(masked_pts[i])
                backprojected_3d_masks["conf"].append(confidences[i])
                backprojected_3d_masks["final_class"].append(labels[i])

            for mask in masked_pts:
                # print("single_mask shape", mask.shape, "all mask shape", masked_counts.shape)
                masked_counts[mask] += 1


        """if no mask is detected"""
        if len(backprojected_3d_masks["conf"]) == 0:
            print("No 3d masks detected in backprojection!")
            # convert to tensor
            backprojected_3d_masks["ins"] = torch.tensor([[]]).to(device=device)  # (Ins, N)
            backprojected_3d_masks["conf"] = torch.tensor([]).to(device=device)  # (Ins, )
            backprojected_3d_masks["final_class"] = []  # (Ins,)
            
            # save the backprojected_3d_masks
            os.makedirs(os.path.join(cfg.mask_3d_dir, text_prompt), exist_ok=True)
            torch.save(
                backprojected_3d_masks,
                os.path.join(cfg.mask_3d_dir, text_prompt, f"{scene_id}.pth"),
            )
            continue
        
        # convert each value in backprojected_3d_masks to tensor
        backprojected_3d_masks["ins"] = torch.stack(
            backprojected_3d_masks["ins"], dim=0
        )  # (Ins, N)
        backprojected_3d_masks["conf"] = torch.tensor(
            backprojected_3d_masks["conf"]
        )  # (Ins,)

        """Aggregating 3d masks"""
        backprojected_3d_masks = aggregate(
            backprojected_3d_masks,
            iou_threshold=cfg.iou_thres,
            feature_similarity_threshold=cfg.similarity_thres,
        )
        
        """if no mask is detected"""
        if len(backprojected_3d_masks["conf"]) == 0:
            print("No 3d masks detected after aggregation")
            # convert to tensor
            backprojected_3d_masks["ins"] = torch.tensor([[]]).to(device=device)  # (Ins, N)
            backprojected_3d_masks["conf"] = torch.tensor([]).to(device=device)  # (Ins, )
            backprojected_3d_masks["final_class"] = []  # (Ins,)
            
            # save the backprojected_3d_masks
            os.makedirs(os.path.join(cfg.mask_3d_dir, text_prompt), exist_ok=True)
            torch.save(
                backprojected_3d_masks,
                os.path.join(cfg.mask_3d_dir, text_prompt, f"{scene_id}.pth"),
            )
            continue

        """Filtering 3d masks"""
        if cfg.if_occurance_threshold:
            occurance_counts = masked_counts.unique()
            # print("occurance count", masked_counts.unique())
            occurance_thres = cfg.occurance_threshold
            occurance_thres_value = occurance_counts[
                round(occurance_thres * occurance_counts.shape[0])
            ]
            # print("occurance thres value", occurance_thres_value)

            # remove all the points under median occurance
            masked_counts[masked_counts < occurance_thres_value] = 0

        elif cfg.if_detected_ratio_threshold:
            # print("DEBUG detected ratio threshold")
            image_dir = os.path.join(scene_2d_dir, scene_id, "color")

            image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
            image_files.sort(
                key=lambda x: int(x.split(".")[0])
            )  # sort numerically, 1.jpg, 2.jpg, 3.jpg ...
            downsampled_image_files = image_files[::cfg.downsample_ratio]  # get one image every 10 frames
            downsampled_images_paths = [
                os.path.join(image_dir, f) for f in downsampled_image_files
            ]

            viewed_counts = torch.zeros(scene_pcd.shape[1]).to(device=device)
            for i, image_path in enumerate(
                tqdm(
                    downsampled_images_paths,
                    desc="Calculating viewed counts for every point",
                    leave=False,
                )
            ):
                frame_id = image_path.split("/")[-1][:-4]
                cam_pose = np.loadtxt(os.path.join(cam_pose_dir, f"{frame_id}.txt"))

                scene_pts = copy.deepcopy(scene_pcd)
                scene_pts = (np.linalg.inv(cam_pose) @ scene_pts).T[:, :3]  # (N, 3)
                projected_pts = compute_projected_pts_tensor(scene_pts, cam_intr)

                photo_width = int(cfg.width_2d)
                photo_height = int(cfg.height_2d)

                # frame_id_num = frame_id.split('.')[0]
                depth_im_path = os.path.join(depth_im_dir, f"{frame_id}.png")
                depth_im = (
                    cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                    / depth_scale
                )
                depth_im = cv2.resize(
                    depth_im, (photo_width, photo_height)
                )  # Width x Height
                visibility_mask = compute_visibility_mask_tensor(
                    scene_pts, projected_pts, depth_im, depth_thresh=0.08
                )
                viewed_counts += torch.tensor(visibility_mask).to(device=device)

            # only calculate non-zero viewed counts
            print("viewed_counts", viewed_counts.unique())
            detected_ratio = masked_counts / (viewed_counts + 1)  # avoid /0
            print("detected_ratio", detected_ratio.unique())
            detected_ratio_thres = cfg.detected_ratio_threshold
            detected_ratio_thres_value = detected_ratio.unique()[
                round(detected_ratio_thres * detected_ratio.unique().shape[0])
            ]
            # print("detected_ratio_thres_value", detected_ratio_thres_value)
            masked_counts[detected_ratio < detected_ratio_thres_value] = 0

        scene_checkpoint[scene_id] = True
        write_scene_checkpoint(text_prompt, scene_checkpoint)

        masked_points = masked_counts > 0  # shape (N,)
        print("final masked points", masked_points.sum())

        # convert each value in backprojected_3d_masks to tensor
        backprojected_3d_masks["ins"] = backprojected_3d_masks["ins"]  # (Ins, N)
        backprojected_3d_masks["conf"] = backprojected_3d_masks["conf"]  # (Ins,)

        # apply filtering on backprojected_3d_masks["ins"]
        print("before filtering", backprojected_3d_masks["ins"].shape)
        num_ins_points_before_filtering = backprojected_3d_masks["ins"].sum(dim=1)  # (Ins,)
        backprojected_3d_masks["ins"] &= masked_points.unsqueeze(0)  # (Ins, N)
        num_ins_points_after_filtering = backprojected_3d_masks["ins"].sum(dim=1)  # (Ins,)
        # print("num_ins_points_before_filtering", num_ins_points_before_filtering)
        # print(" num of points being filtered in each mask", num_ins_points_before_filtering-num_insgi_points_after_filtering)

        # delete the masks with less than 1/2 points after filtering and have more than 50 points
        backprojected_3d_masks["ins"] = backprojected_3d_masks["ins"][
            (num_ins_points_after_filtering > cfg.remove_small_masks)
            & (
                num_ins_points_after_filtering
                > cfg.remove_filtered_masks * num_ins_points_before_filtering
            )
        ]
        backprojected_3d_masks["conf"] = backprojected_3d_masks["conf"].to(device=device)
        # also delete the corresponding confidences and labels
        backprojected_3d_masks["conf"] = backprojected_3d_masks["conf"][
            (num_ins_points_after_filtering > cfg.remove_small_masks)
            & (
                num_ins_points_after_filtering
                > cfg.remove_filtered_masks * num_ins_points_before_filtering
            )
        ]
        backprojected_3d_masks["final_class"] = [
            backprojected_3d_masks["final_class"][i]
            for i in range(len(backprojected_3d_masks["final_class"]))
            if num_ins_points_after_filtering[i] > cfg.remove_small_masks
            and num_ins_points_after_filtering[i]
            > cfg.remove_filtered_masks * num_ins_points_before_filtering[i]
        ]
        
        print("after filtering", backprojected_3d_masks["ins"].shape)
        print("num_ins_points_after_filtering", backprojected_3d_masks["ins"].sum(dim=1))

        # save the backprojected_3d_masks
        os.makedirs(os.path.join(cfg.mask_3d_dir, text_prompt), exist_ok=True)
        torch.save(
            backprojected_3d_masks,
            os.path.join(cfg.mask_3d_dir, text_prompt, f"{scene_id}.pth"),
        )
