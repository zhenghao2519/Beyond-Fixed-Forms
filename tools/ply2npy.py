import numpy as np
import open3d as o3d
import os


def ply2npy(ply_path: str, npy_path: str):
    # read ply file
    pc = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pc.points)
    colors = np.asarray(pc.colors)
    print("points shape", points.shape, "colors", colors.shape)

    # mean_color = np.mean(colors, axis = 0)
    # print("mean color", mean_color)

    output = np.concatenate([points, colors], axis=1)
    print("output shape", output.shape)
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    np.save(npy_path,output)


if __name__ == "__main__":
    ply_path = "./data/Scannet200/Scannet200_3D/val/original_ply_files/scene0435_00.ply"
    npy_path = "./data/Scannet200/Scannet200_3D/val/original_npy_files/scene0435_00.npy"
    ply2npy(ply_path, npy_path)