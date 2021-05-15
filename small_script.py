import open3d as o3d
import numpy as np
from downsample import load_bin_file
from utils import draw_geometries_dark_background

# def visualize(filepath):




pcl_dict = {"full": "original/",
            "5": "voxelsize_0.8100/",
            "7.5": "voxelsize_0.5950/",
            "10": "voxelsize_0.4750/",
            "20": "voxelsize_0.2600/",
            "25": "voxelsize_0.2100/",
            "30": "voxelsize_0.1740/",
            "40": "voxelsize_0.1256/",
            "50": "voxelsize_0.0940/",
            "60": "voxelsize_0.0708/",
            "70": "voxelsize_0.0535/",
            "75": "voxelsize_0.0462/",
            "80": "voxelsize_0.0394/",
            "90": "voxelsize_0.0271/",
            "every_2": "everyk_2/",
            "every_3": "everyk_3/",
            "every_4": "everyk_4/",
            "every_5": "everyk_5/",
            "every_10": "everyk_10/",
            "every_20": "everyk_20/"}


if __name__ == "__main__":
    lidar_folder = "/home/mrsd2/Documents/vlr-project/Voxel-R-CNN/data/kitti/training/all_velodynes/"
    bin_file = "000351.bin"


    ds_folder = pcl_dict["10"] + "velodyne/"
    filepath = lidar_folder + ds_folder + bin_file
    o3d_pcd = load_bin_file(filepath)
    o3d_pcd.paint_uniform_color([0.65, 0.65, 0.65])
    draw_geometries_dark_background([o3d_pcd], "lidar.png")

    ds_folder1 = pcl_dict["every_10"] + "velodyne/"
    filepath1 = lidar_folder + ds_folder1 + bin_file
    o3d_pcd1 = load_bin_file(filepath1)
    o3d_pcd1.paint_uniform_color([0.65, 0.65, 0.65])
    draw_geometries_dark_background([o3d_pcd1], "lidar.png")

    # vis = o3d.visualization.Visualizer()
    # vis.add_geometry(o3d_pcd)
    # opt = vis.get_render_option()
    # opt.background_color = np.asarray([0, 0, 0])
    # o3d.visualization.draw_geometries([o3d_pcd], zoom=0.3412,
    #                                   front=[0.4257, -0.2125, -0.8795],
    #                                   lookat=[2.6172, 2.0475, 1.532],
    #                                   up=[-0.0694, -0.9768, 0.2024])

