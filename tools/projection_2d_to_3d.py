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
import sys

sys.path.append("/medar_smart/temp/Beyond-Fixed-Forms/tools")
from segmentation_2d import inference_grounded_sam

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# def compute_projected_pts(pts, cam_intr):
#     # map 3d pointclouds in camera coordinates system to 2d
#     N = pts.shape[0]
#     projected_pts = np.empty((N, 2), dtype=np.int64)
#     fx, fy = cam_intr[0, 0], cam_intr[1, 1]
#     cx, cy = cam_intr[0, 2], cam_intr[1, 2]
#     print("pt0 " ,pts[0])
#     for i in range(pts.shape[0]):
#         z = pts[i, 2]
#         x = int(np.round(fx * pts[i, 0] / z + cx))
#         y = int(np.round(fy * pts[i, 1] / z + cy))
#         projected_pts[i, 0] = x
#         projected_pts[i, 1] = y
#     print("projected_pts", projected_pts[0])
#     return projected_pts


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


# def compute_visibility_mask(pts, projected_pts, depth_im, depth_thresh=0.005):
#     # compare z in camera coordinates and depth image
#     # to check if there projected points are visible
#     im_h, im_w = depth_im.shape
#     # print(depth_im.shape)
#     visibility_mask = np.zeros(projected_pts.shape[0]).astype(np.bool8)
#     z = pts[:, 2]
#     # print("DEBUG z value", z.max(), z.min(),z.mean())
#     for i in range(projected_pts.shape[0]):
#         x, y = projected_pts[i]
#         z = pts[i, 2]
#         if x < 0 or x >= im_w or y < 0 or y >= im_h:
#             continue
#         if depth_im[y, x] == 0:
#             continue
#         if np.abs(z - depth_im[y, x]) < depth_thresh and z > 0:
#             visibility_mask[i] = True
#     return visibility_mask


def compute_visibility_mask_tensor(pts, projected_pts, depth_im, depth_thresh=0.005):
    # compare z in camera coordinates and depth image
    # to check if there projected points are visible
    im_h, im_w = depth_im.shape

    visibility_mask = np.zeros(projected_pts.shape[0]).astype(np.bool8)
    inbounds = (
        (projected_pts[:, 0] >= 0) &
        (projected_pts[:, 0] < im_w) &
        (projected_pts[:, 1] >= 0) &
        (projected_pts[:, 1] < im_h)
    )   # (N,)
    projected_pts = projected_pts[inbounds]  # (X, 2)
    depth_check = (
        depth_im[projected_pts[:, 1], projected_pts[:, 0]] != 0
    ) & (
        np.abs(pts[inbounds][:, 2] - depth_im[projected_pts[:, 1], projected_pts[:, 0]])
        < depth_thresh
    )

    visibility_mask[inbounds] = depth_check
    return visibility_mask # (N,)


# def compute_visible_masked_pts(scene_pts, projected_pts, visibility_mask, pred_masks):
#     # return masked 3d points
#     N = scene_pts.shape[0]
#     M, _, _ = pred_masks.shape  # (M, H, W)
#     # print("DEBUG M value", M)
#     masked_pts = np.zeros((M, N), dtype=np.bool_)
#     visible_indices = np.nonzero(visibility_mask)[0]
#     for m in range(M):
#         for i in visible_indices:
#             x, y = projected_pts[i]
#             if pred_masks[m, y, x]:
#                 masked_pts[m, i] = True
#     return masked_pts

def compute_visible_masked_pts_tensor(scene_pts, projected_pts, visibility_mask, pred_masks):
    # return masked 3d points
    N = scene_pts.shape[0]
    M, _, _ = pred_masks.shape  # (M, H, W)
    # print("DEBUG M value", M)
    masked_pts = np.zeros((M, N), dtype=np.bool_)
    visiable_pts = projected_pts[visibility_mask] # (X, 2)
    for m in range(M):
        x, y = visiable_pts.T # (X,)
        mask_check = pred_masks[m, y, x] # (X,)
        masked_pts[m, visibility_mask] = mask_check
    
    return masked_pts

# def rle_decode(rle):
#     """
#     Decode rle to get binary mask.
#     Args:
#         rle (dict): rle of encoded mask
#     Returns:
#         mask (np.ndarray): decoded mask
#     """
#     length = rle["length"]
#     try:
#         s = rle["counts"].split()
#     except:
#         s = rle["counts"]

#     starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
#     starts -= 1
#     ends = starts + nums
#     mask = np.zeros(length, dtype=np.uint8)
#     for lo, hi in zip(starts, ends):
#         mask[lo:hi] = 1
#     return mask


def get_parser():
    parser = argparse.ArgumentParser(description="Configuration Open3DIS")
    parser.add_argument("--config", type=str, required=True, help="Config")
    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()
    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))
    scene_id = cfg.scene_id
    mask_2d_dir = cfg.mask_2d_dir
    cam_pose_dir = cfg.cam_pose_dir
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # N = 254998 # number of 3d points
    # visibility_count = np.zeros((N,), dtype=np.int32)

    # load camera configs
    # cam_intr = np.loadtxt("/home/jie_zhenghao/Open3DIS/data/Scannet200/Scannet200_2D_5interval/val/scene0435_00/intrinsic.txt")
    # cam_intr =np.array(
    #             [[577.870605, 0.0, 319.5],
    #              [0.0, 577.870605, 239.5],
    #              [0.0, 0.0, 1.0]])
    cam_intr = np.array(
        [[1170.1, 0.0, 647.7], [0.0, 1170.187988, 483.750000], [0.0, 0.0, 1.0]]
    )
    # print(cam_intr)
    # print(cam_intr.shape)
    # cam_intr = np.asarray(cam_config['cam_intr'])
    depth_scale = 1000  # not sure, JUST TRYING!

    # load 3d point cloud
    scene_pcd_path = cfg.scene_pcd_path
    # scene_pcd = o3d.io.read_point_cloud(str(scene_pcd_path))
    scene_pcd = np.load(scene_pcd_path)[:, :3]
    scene_pcd = np.concatenate(
        [scene_pcd, torch.ones([scene_pcd.shape[0], 1])], axis=1
    ).T
    # print("scene_pcd shape:", scene_pcd.shape)

    # 2d sam masks
    # annotated_frame, segmented_frame_masks = inference_grounded_sam()
    masks_2d_path = os.path.join(mask_2d_dir, f"{scene_id}.pth")
    gronded_sam_results = torch.load(masks_2d_path)

    masked_counts = torch.zeros(scene_pcd.shape[1]).to(
        device=device
    )  # scene_pcd has shape (4, N)
    for i in range(len(gronded_sam_results)):  # range(35,40):

        frame_id = gronded_sam_results[i]["frame_id"][:-4]
        print("-------------------------frame", frame_id, "-------------------------")
        segmented_frame_masks = gronded_sam_results[i]["segmented_frame_masks"]

        pred_masks = segmented_frame_masks.squeeze(dim=1).numpy()  # (M, H, W)
        cam_pose = np.loadtxt(os.path.join(cam_pose_dir, f"{frame_id}.txt"))

        scene_pts = copy.deepcopy(scene_pcd)
        scene_pts = (np.linalg.inv(cam_pose) @ scene_pts).T[:, :3]  # (N, 3)
        projected_pts = compute_projected_pts_tensor(scene_pts, cam_intr)

        photo_width = int(cfg.width_2d)
        photo_height = int(cfg.height_2d)
        depth_im_dir = cfg.depth_im_dir
        # frame_id_num = frame_id.split('.')[0]
        depth_im_path = os.path.join(cfg.depth_im_dir, f"{frame_id}.png")
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


        masked_pts = torch.from_numpy(masked_pts).to(device)
        mask_area = torch.sum(masked_pts, dim=1).detach().cpu().numpy()  # (M,)
        print(
            "number of 3d mask points:",
            mask_area,
            "number of 2d masks:",
            pred_masks.sum(axis=(1, 2)),
        )

        for mask in masked_pts:
            # print("single_mask shape", mask.shape, "all mask shape", masked_counts.shape)
            masked_counts[mask] += 1


    ## Start filtering
    if cfg.if_occurance_threshold:
        occurance_counts = masked_counts.unique()
        print("occurance count", masked_counts.unique())
        occurance_thres = cfg.occurance_threshold
        occurance_thres_value = occurance_counts[
            round(occurance_thres * occurance_counts.shape[0])
        ]
        print("occurance thres value", occurance_thres_value)

        # remove all the points under median occurance
        masked_counts[masked_counts < occurance_thres_value] = 0

    elif cfg.if_detected_ratio_threshold:
        scene_id = cfg.scene_id
        root_dir = cfg.root_dir
        image_dir = os.path.join(root_dir, scene_id, "color")

        image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        image_files.sort(key=lambda x: int(x.split('.')[0]))         # sort numerically, 1.jpg, 2.jpg, 3.jpg ...
        downsampled_image_files = image_files[::10]                  # get one image every 10 frames
        downsampled_images_paths = [os.path.join(image_dir, f) for f in downsampled_image_files]

        viewed_counts = torch.zeros(scene_pcd.shape[1]).to(device=device)
        for i, image_path in enumerate(tqdm(downsampled_images_paths, desc="Calculating viewed counts for every point", leave=False)):
            frame_id = image_path.split('/')[-1][:-4]
            cam_pose = np.loadtxt(os.path.join(cam_pose_dir, f"{frame_id}.txt"))

            scene_pts = copy.deepcopy(scene_pcd)
            scene_pts = (np.linalg.inv(cam_pose) @ scene_pts).T[:, :3]  # (N, 3)
            projected_pts = compute_projected_pts_tensor(scene_pts, cam_intr)

            photo_width = int(cfg.width_2d)
            photo_height = int(cfg.height_2d)
            depth_im_dir = cfg.depth_im_dir
            # frame_id_num = frame_id.split('.')[0]
            depth_im_path = os.path.join(cfg.depth_im_dir, f"{frame_id}.png")
            depth_im = (
                cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                / depth_scale
            )
            depth_im = cv2.resize(depth_im, (photo_width, photo_height))  # Width x Height
            visibility_mask = compute_visibility_mask_tensor(
                scene_pts, projected_pts, depth_im, depth_thresh=0.08
            )
            viewed_counts += torch.tensor(visibility_mask).to(device=device)

        # only calculate non-zero viewed counts
        print("viewed_counts", viewed_counts.unique())
        detected_ratio = masked_counts / (viewed_counts+1) # avoid /0
        print("detected_ratio", detected_ratio.unique())
        detected_ratio_thres = cfg.detected_ratio_threshold
        detected_ratio_thres_value = detected_ratio.unique()[round(detected_ratio_thres * detected_ratio.unique().shape[0])]
        print("detected_ratio_thres_value", detected_ratio_thres_value)
        masked_counts[detected_ratio < detected_ratio_thres_value] = 0





    masked_counts = masked_counts > 0
    print("final masked points", masked_counts.sum())

    os.makedirs(os.path.join(cfg.mask_3d_dir, cfg.base_prompt), exist_ok=True)
    torch.save(
        {
            "ins": masked_counts.unsqueeze(0),
            "conf": torch.tensor([0.36]),
            "final_class": torch.tensor([30]),
        },
        os.path.join(cfg.mask_3d_dir, cfg.base_prompt, f"{scene_id}.pth"),
    )
