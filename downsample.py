import numpy as np
import open3d as o3d
import os
import multiprocessing
from itertools import repeat
from datetime import datetime


def print_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(current_time)
    return None


def downsample(lidar_file, input_lidar_dir, output_lidar_dir, factor, method="voxel"):
    # Helper function to run downsampling on a directory of lidar files
    lidar_path = input_lidar_dir + lidar_file
    point_cloud, orig_points = load_bin_file(lidar_path, return_num_points=True)

    if method == "voxel":  # Choose downsampling method
        downpcd = point_cloud.voxel_down_sample(voxel_size=factor)  # Use voxels of size factor
    elif method == "every_k":
        downpcd = point_cloud.uniform_down_sample(every_k_points=factor)  # Keep every_k_points

    ds_points = save_as_bin_file(downpcd, output_lidar_dir + lidar_file, return_num_points=True)
    return orig_points, ds_points

def downsample_files(input_lidar_dir, output_lidar_dir, factor, method="voxel"):
    # Downsample a directory of .bin files and save them in another directory; method = ["voxel", "every_k"]
    # No multithreading
    print("Starting to downsample files.")
    print_time()

    if method == "voxel":
        output_lidar_dir += "voxelsize_" + str(factor) + "/velodyne/"
    elif method == "every_k":
        output_lidar_dir += "everyk_" + str(factor) + "/velodyne/"

    if not os.path.exists(output_lidar_dir):
        os.makedirs(output_lidar_dir)
    lidar_files = os.listdir(input_lidar_dir)
    lidar_files.sort()
    ds_pts_total, orig_pts_total = 0, 0

    for i, lidar_file in enumerate(lidar_files):
        orig_points, ds_points = downsample(lidar_file, input_lidar_dir, output_lidar_dir, factor, method)
        ds_pts_total += ds_points
        orig_pts_total += orig_points
        if i % 500 == 0 and i != 0:
            print("Processed " + str(i) + " files.")

    fraction = ds_pts_total / orig_pts_total
    percent = round(fraction*100, 1)  # Convert to percent and keep 1 decimal point

    np.savetxt(output_lidar_dir.replace("velodyne/", "") + str(percent) + "%" + ".txt", np.array([percent]))
    print("Finished downsampling.")
    print_time()
    print("Percent of original point cloud: ", percent)
    return fraction


def downsample_files_mp(input_lidar_dir, output_lidar_dir, factor, method="voxel", num_workers=8):
    # Downsample a directory of .bin files and save them in another directory; method = ["voxel", "every_k"]
    # Uses multithreading
    print("Starting to downsample files.")
    print_time()

    if method == "voxel":
        output_lidar_dir += "voxelsize_" + str(factor) + "/velodyne/"
    elif method == "every_k":
        output_lidar_dir += "everyk_" + str(factor) + "/velodyne/"

    if not os.path.exists(output_lidar_dir):
        os.makedirs(output_lidar_dir)  # Create directory if not already existing
    lidar_files = os.listdir(input_lidar_dir)  # Get list of bin files and sort
    lidar_files.sort()

    # # Multithreading
    pool = multiprocessing.Pool(processes=num_workers)
    result = pool.starmap(downsample, zip(lidar_files, repeat(input_lidar_dir), repeat(output_lidar_dir), repeat(factor), repeat(method)))

    pts = np.array(result).sum(axis=0)  # Result is a list of tuples of the output from the multithreaded function
    fraction = pts[1] / pts[0]

    percent = round(fraction*100, 1)  # Convert to percent and keep 1 decimal point

    np.savetxt(output_lidar_dir.replace("velodyne/", "") + str(percent) + "%" + ".txt", np.array([percent]))
    print("Finished downsampling.")
    print_time()
    print("Percent of original point cloud: ", percent)
    return fraction


def load_bin_file(file, return_num_points=False): 
    # Takes a binary file and returns an open3D point cloud object and the number of points in the cloud
    file_data = np.fromfile(file, dtype=np.float32).reshape(-1, 4)  # (x, y, z, reflectance)
    point_cloud = file_data[:, 0:3]  # Get (x, y, z) points
    reflectance = file_data[:, -1]  # Get reflectance values
    colors = np.hstack(([np.expand_dims(reflectance, axis=1)] * 3))  # Save the reflectance as colors in the point cloud obj; need to store as RGB triplet

    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud))  # Convert arrays to open3d vectors
    o3d_colors = o3d.utility.Vector3dVector(colors)
    o3d_pcd.colors = o3d_colors
    num_points = len(point_cloud)
    if return_num_points == True:
        return o3d_pcd, num_points
    return o3d_pcd


def save_as_bin_file(o3d_pcd, save_name, return_num_points=False):
    # Input an Open3D point cloud object and save as a binary file
    new_colors = np.expand_dims(np.asarray(o3d_pcd.colors, dtype=np.float32)[:, 0], axis=1)  # Get back reflectance values from downsampled point cloud
    np_pts = np.asarray(o3d_pcd.points, dtype=np.float32)  # Change open3d array to np array
    np_pts_all = np.hstack((np_pts, new_colors))  # Stack points
    np_pts_flatten = np_pts_all.flatten()  # Flatten points before saving
    np_pts_flatten.tofile(save_name)
    num_points = len(np_pts)
    if ".bin" not in save_name:
        save_name = save_name + ".bin"
    # print("Saved point cloud data to", save_name)
    if return_num_points == True:
        return num_points
    return None


if __name__ == "__main__":
    # Directory setups
    project_dir = "/home/mrsd2/Documents/vlr-project/Voxel-R-CNN/"
    data_dir = project_dir + "data/kitti/training/"
    lidar_dir = data_dir + "all_velodynes/original/velodyne/"
    label_dir = data_dir + "label_2/"
    lidar_downsample_dir = data_dir + "all_velodynes/"  # Don't change unless all_velodynes folder changes

    # Downsample a folder of point clouds
    num_workers = 8  # For multithreading
    factor = 10  # Integer for every k, voxel size for voxel
    # method = ["every_k", "voxel"])
    fraction = downsample_files_mp(lidar_dir, lidar_downsample_dir, factor=factor, method="every_k", num_workers=num_workers)


    def test2():
        # Test compression ratio of points only in front of the camera
        orig_lidar_path = "/home/mrsd2/Documents/vlr-project/Voxel-R-CNN/data/kitti/training/all_velodynes/original/velodyne/"
        ds_lidar_path = "/home/mrsd2/Documents/vlr-project/Voxel-R-CNN/data/kitti/training/all_velodynes/voxelsize_0.0940/velodyne/"  # 50%

        orig_lidar_files = os.listdir(orig_lidar_path)
        ds_lidar_files = os.listdir((ds_lidar_path))
        orig_lidar_files.sort(), ds_lidar_files.sort()
        ds_pts, orig_pts = 0, 0
        for i in range(len(orig_lidar_files)):
            point_cloud_orig = np.fromfile(orig_lidar_path + orig_lidar_files[0], dtype=np.float32).reshape(-1, 4)
            point_cloud_ds = np.fromfile(ds_lidar_path + orig_lidar_files[0], dtype=np.float32).reshape(-1, 4)

            # remove points that are located behind the camera:
            point_cloud_orig = point_cloud_orig[point_cloud_orig[:, 0] > -2.5, :]
            point_cloud_ds = point_cloud_ds[point_cloud_ds[:, 0] > -2.5, :]

            orig_pts += point_cloud_orig.shape[0]
            ds_pts += point_cloud_ds.shape[0]

        fraction = ds_pts / orig_pts
        print(fraction)
        return None


    def test1():
        # Test the loading, downsampling, and saving of a point cloud from a .bin file
        img_id = "000001"
        lidar_path = lidar_dir + img_id + ".bin"
        label_path = label_dir + img_id + ".txt"

        o3d_pcd = load_bin_file(lidar_path)

        o3d.visualization.draw_geometries([o3d_pcd],
                                        zoom=0.3412,
                                        front=[0.4257, -0.2125, -0.8795],
                                        lookat=[2.6172, 2.0475, 1.532],
                                        up=[-0.0694, -0.9768, 0.2024])


        o3d.visualization.draw_geometries([downpcd],
                                        zoom=0.3412,
                                        front=[0.4257, -0.2125, -0.8795],
                                        lookat=[2.6172, 2.0475, 1.532],
                                        up=[-0.0694, -0.9768, 0.2024])


        test_save_file = "ds_000001.bin"
        save_as_bin_file(downpcd, test_save_file)
        pcd_reloaded = load_bin_file(test_save_file)

        o3d.visualization.draw_geometries([pcd_reloaded],
                                        zoom=0.3412,
                                        front=[0.4257, -0.2125, -0.8795],
                                        lookat=[2.6172, 2.0475, 1.532],
                                        up=[-0.0694, -0.9768, 0.2024])

        return print("Yee")

