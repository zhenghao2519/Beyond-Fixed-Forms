import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import copy
import torch

import sys
sys.path.append('/home/jie_zhenghao/Beyond-Fixed-Forms/tools')
from segmentation_2d import inference_grounded_sam

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
    print(depth_im.shape)
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
        if np.abs(z - depth_im[y, x]) < depth_thresh:
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


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() 
                          else 'mps' if torch.backends.mps.is_available() 
                          else 'cpu')

    N = 254998 # number of 3d points
    visibility_count = np.zeros((N,), dtype=np.int32)

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
    print(cam_intr)
    print(cam_intr.shape)
    # cam_intr = np.asarray(cam_config['cam_intr'])
    depth_scale = 1000 # not sure, JUST TRYING!


    # load 3d point cloud
    scene_pcd_path = "/home/jie_zhenghao/Open3DIS/data/Scannet200/Scannet200_3D/val/original_ply_files/scene0435_00.ply"
    scene_pcd = o3d.io.read_point_cloud(str(scene_pcd_path))


    # 2d sam masks
    image_path = '../data/Scannet200/Scannet200_2D_5interval/val/scene0435_00/color/738.jpg' #"assets/demo9.jpg" 738
    base_prompt = "Clothes, Pillow, Chair, Sofa, Bed, Desk, Monitor, Television, Book"
    annotated_frame, detected_boxes, segmented_frame_masks = inference_grounded_sam(image_path, base_prompt)
    pred_masks = segmented_frame_masks.squeeze(dim=1).numpy()  # (M, H, W)
    print(pred_masks.shape, pred_masks.sum(axis=(1,2)), pred_masks.max())

    # map 3d pointclouds to 2d
    cam_pose = np.loadtxt("/home/jie_zhenghao/Open3DIS/data/Scannet200/Scannet200_2D_5interval/val/scene0435_00/pose/738.txt")

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



    pcd = copy.deepcopy(scene_pcd).transform(np.linalg.inv(cam_pose)) # world to camera
    scene_pts = np.asarray(pcd.points)
    projected_pts = compute_projected_pts(scene_pts, cam_intr)
    print("projected_pts", projected_pts[:,0].mean(), projected_pts[:,1].mean())

    depth_im_path = "/home/jie_zhenghao/Open3DIS/data/Scannet200/Scannet200_2D_5interval/val/scene0435_00/depth/738.png"
    depth_im = cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32)/depth_scale
    print(depth_im.shape)
    plt.imshow(depth_im, cmap='gray')
    plt.colorbar(label='Depth (meters)')
    plt.title('Depth Image Visualization')
    plt.show()
    # depth_im = depth_im/depth_scale
    visibility_mask = compute_visibility_mask(scene_pts, projected_pts, depth_im, depth_thresh=0.8)
    # visibility_count[visibility_mask] += 1
    print(depth_im.max())
    print(visibility_mask.sum())

    masked_pts = compute_visible_masked_pts(scene_pts, projected_pts, visibility_mask, pred_masks)  # (M, N)
    masked_pts = torch.from_numpy(masked_pts).to(device)
    # if ground_indices is not None:
    #     masked_pts[:, ground_indices] = 0
    mask_area = torch.sum(masked_pts, dim=1).detach().cpu().numpy()  # (M,)