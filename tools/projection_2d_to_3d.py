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

import sys
sys.path.append('/medar_smart/temp/Beyond-Fixed-Forms/tools')
from segmentation_2d import inference_grounded_sam

device = torch.device('cuda' if torch.cuda.is_available() 
                          else 'mps' if torch.backends.mps.is_available() 
                          else 'cpu')

def compute_projected_pts(pts, cam_intr):
    # map 3d pointclouds in camera coordinates system to 2d
    N = pts.shape[0]
    projected_pts = np.empty((N, 2), dtype=np.int64)
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]
    for i in range(pts.shape[0]):
        z = pts[i, 2]
        x = int(np.round(fx * pts[i, 0] / z + cx))
        y = int(np.round(fy * pts[i, 1] / z + cy))
        projected_pts[i, 0] = x
        projected_pts[i, 1] = y
    return projected_pts


def compute_visibility_mask(pts, projected_pts, depth_im, depth_thresh=0.005):
    # compare z in camera coordinates and depth image 
    # to check if there projected points are visible
    im_h, im_w = depth_im.shape
    # print(depth_im.shape)
    visibility_mask = np.zeros(projected_pts.shape[0]).astype(np.bool8)
    z = pts[:, 2]
    # print("DEBUG z value", z.max(), z.min(),z.mean())
    for i in range(projected_pts.shape[0]):
        x, y = projected_pts[i]
        z = pts[i, 2]
        if x < 0 or x >= im_w or y < 0 or y >= im_h:
            continue
        if depth_im[y, x] == 0:
            continue
        if np.abs(z - depth_im[y, x]) < depth_thresh and z > 0:
            visibility_mask[i] = True
    return visibility_mask


def compute_visible_masked_pts(scene_pts, projected_pts, visibility_mask, pred_masks):
    # return masked 3d points
    N = scene_pts.shape[0]
    M, _, _ = pred_masks.shape  # (M, H, W)
    # print("DEBUG M value", M)
    masked_pts = np.zeros((M, N), dtype=np.bool_)
    visible_indices = np.nonzero(visibility_mask)[0]
    for m in range(M):
        for i in visible_indices:
            x, y = projected_pts[i]
            if pred_masks[m, y, x]:
                masked_pts[m, i] = True
    return masked_pts

def rle_decode(rle):
    """
    Decode rle to get binary mask.
    Args:
        rle (dict): rle of encoded mask
    Returns:
        mask (np.ndarray): decoded mask
    """
    length = rle["length"]
    try:
        s = rle["counts"].split()
    except:
        s = rle["counts"]

    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask

def get_parser():
    parser = argparse.ArgumentParser(description="Configuration Open3DIS")
    parser.add_argument("--config",type=str,required = True,help="Config")
    return parser

if __name__ == "__main__":
    
    args = get_parser().parse_args()
    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))
    scene_id = cfg.scene_id
    mask_2d_dir = cfg.mask_2d_dir
    cam_pose_dir = cfg.cam_pose_dir
    device = torch.device('cuda' if torch.cuda.is_available() 
                          else 'mps' if torch.backends.mps.is_available() 
                          else 'cpu')

    # N = 254998 # number of 3d points
    # visibility_count = np.zeros((N,), dtype=np.int32)

    # load camera configs
    # cam_intr = np.loadtxt("/home/jie_zhenghao/Open3DIS/data/Scannet200/Scannet200_2D_5interval/val/scene0435_00/intrinsic.txt")
    # cam_intr =np.array(
    #             [[577.870605, 0.0, 319.5], 
    #              [0.0, 577.870605, 239.5],
    #              [0.0, 0.0, 1.0]])
    cam_intr =np.array(
                [[1170.1, 0.0, 647.7], 
                [0.0, 1170.187988, 483.750000],
                [0.0, 0.0, 1.0]])
    # print(cam_intr)
    # print(cam_intr.shape)
    # cam_intr = np.asarray(cam_config['cam_intr'])
    depth_scale = 1000 # not sure, JUST TRYING!


    # load 3d point cloud
    scene_pcd_path = cfg.scene_pcd_path
    # scene_pcd = o3d.io.read_point_cloud(str(scene_pcd_path))
    scene_pcd = np.load(scene_pcd_path)[:,:3]
    scene_pcd = np.concatenate([scene_pcd, torch.ones([scene_pcd.shape[0], 1])], axis=1).T
    # print("scene_pcd shape:", scene_pcd.shape)

    # 2d sam masks
    # annotated_frame, segmented_frame_masks = inference_grounded_sam()
    masks_2d_path = os.path.join(mask_2d_dir, f"{scene_id}.pth")
    gronded_sam_results = torch.load(masks_2d_path) 

    all_points_masked = torch.zeros(scene_pcd.shape[1]) # scene_pcd has shape (4, N)
    for i in range(len(gronded_sam_results)): # range(35,40): 

        frame_id = gronded_sam_results[i]["frame_id"][:-4]
        print("-------------------------frame", frame_id, "-------------------------")
        segmented_frame_masks = gronded_sam_results[i]["segmented_frame_masks"]
        # segmented_frame_masks = rle_decode(gronded_sam_results[i]["segmented_frame_masks_rle"])    # TODO: HERE IS THE INFERENCE RESULTS OF ALL FRAMES (A DICTIONARY)

        pred_masks = segmented_frame_masks.squeeze(dim=1).numpy()  # (M, H, W)
        # print(pred_masks.shape, pred_masks.sum(axis=(1,2)), pred_masks.max())
        # map 3d pointclouds to 2d
        cam_pose = np.loadtxt(os.path.join(cam_pose_dir, f"{frame_id}.txt"))

        ###!!!!!!!!! TEST IF TRANSFORMED CORRECTLY
        # coords = torch.from_numpy(np.asarray(scene_pcd.points))
        # print("shape of coords",coords.shape, coords.max())
        # coords_new = torch.cat([coords, torch.ones([coords.shape[0], 1], dtype=torch.float, device='cpu')], dim=1).T
        # print("shape of coords_new",coords_new.shape, coords_new.max())
        # world_to_camera = torch.linalg.inv(torch.from_numpy(cam_pose))
        # print("shape of world_to_camera",world_to_camera)
        # p = world_to_camera.float() @ coords_new.float()
        # print("shape of p after mult",p.shape, p.max())
        # cam_intr_t = torch.from_numpy(np.array(
        #             [[577.870605, 0.0, 319.5], 
        #              [0.0, 577.870605, 239.5],
        #              [0.0, 0.0, 1.0]]))
        # p[0] = (p[0] * cam_intr_t[0][0]) / p[2] + cam_intr_t[0][2]
        # p[1] = (p[1] * cam_intr_t[1][1]) / p[2] + cam_intr_t[1][2]
        # # p = p.reshape(-1,4)
        # print("shape of p final p",p.shape, p[0][0])

        scene_pts = copy.deepcopy(scene_pcd)
        scene_pts = (np.linalg.inv(cam_pose) @ scene_pts).T[:,:3] # (N, 3)
        # print(scene_pts.shape, np.max(scene_pts[:,2]), np.min(scene_pts[:,2]))
        # pcd = copy.deepcopy(pcd).transform(np.linalg.inv(cam_pose)) # world to camera
        # scene_pts = np.asarray(pcd.points)
        projected_pts = compute_projected_pts(scene_pts, cam_intr)
        # print("projected_pts", projected_pts[:,0].mean(), projected_pts[:,1].mean())

        depth_im_dir = cfg.depth_im_dir
        # frame_id_num = frame_id.split('.')[0]
        depth_im_path = os.path.join(cfg.depth_im_dir, f"{frame_id}.png")
        depth_im = cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32)/depth_scale
        depth_im = cv2.resize(depth_im, (1296, 968)) # Width x Height
        # print("depth image shape:" , depth_im.shape, "mask shape",  pred_masks.shape)
        # plt.imshow(depth_im, cmap='gray')
        # plt.colorbar(label='Depth (meters)')
        # plt.title('Depth Image Visualization')
        # plt.show()
        # # depth_im = depth_im/depth_scale
        visibility_mask = compute_visibility_mask(scene_pts, projected_pts, depth_im, depth_thresh=0.08)
        # visibility_count[visibility_mask] += 1
        # print(depth_im.max())
        # print("visibility check result", visibility_mask.sum())

        masked_pts = compute_visible_masked_pts(scene_pts, projected_pts, visibility_mask, pred_masks)  # (M, N)
        masked_pts = torch.from_numpy(masked_pts).to(device)
        # if ground_indices is not None:
        #     masked_pts[:, ground_indices] = 0
        mask_area = torch.sum(masked_pts, dim=1).detach().cpu().numpy()  # (M,)
        print("number of 3d mask points:", mask_area, "number of 2d masks:", pred_masks.sum(axis=(1,2)))
        
        for mask in masked_pts:
            # print("single_mask shape", mask.shape, "all mask shape", all_points_masked.shape)
            all_points_masked[mask] = 1
          
    print("final masked points", all_points_masked.sum())  
    if not os.path.exists(cfg.mask_3d_dir):
        os.makedirs(cfg.mask_3d_dir)
    torch.save({'ins':all_points_masked.unsqueeze(0), 'conf': torch.tensor([0.36]), 'final_class': torch.tensor([30])}, os.path.join(cfg.mask_3d_dir, f"{scene_id}.pth"))
        
