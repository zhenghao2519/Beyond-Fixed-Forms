import numpy as np
import open3d as o3d
import os
from tqdm import tqdm


def ply2npy(ply_path: str, npy_path: str):
    # read ply file
    pc = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pc.points)
    colors = np.asarray(pc.colors)
    print(f"Processing {ply_path}: points shape: {points.shape}, colors: {colors.shape}")

    # mean_color = np.mean(colors, axis = 0)
    # print("mean color", mean_color)

    output = np.concatenate([points, colors], axis=1)
    print("output shape", output.shape)
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    np.save(npy_path,output)

def process_all_ply_files(ply_dir: str, npy_dir: str):
    # Ensure the output directory exists
    os.makedirs(npy_dir, exist_ok=True)
    
    # Process each ply file in the directory
    for scene in tqdm(os.listdir(ply_dir), desc="Converting .ply to .npy"):
        if scene.endswith('.ply'):
            ply_path = os.path.join(ply_dir, scene)
            npy_path = os.path.join(npy_dir, scene.replace('.ply', '.npy'))
            ply2npy(ply_path, npy_path)


if __name__ == "__main__":
    ply_path = "./data/Scannet200/Scannet200_3D/original_ply_files"
    npy_path = "./data/Scannet200/Scannet200_3D/original_npy_files"
    process_all_ply_files(ply_path, npy_path)