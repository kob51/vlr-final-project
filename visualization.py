#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 17:48:51 2021

@author: mrsd2
"""
# camera-ready

import pickle
import numpy as np
import cv2
import open3d
from utils import *


# REFERENCE CODE #########################################
# https://github.com/fregu856/3DOD_thesis
##########################################################

# DATA SETUP ###############################################################
img_height = 375
img_width = 1242

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

# USER INPUTS #####################################################################
start_frame = 168
end_frame = 169    # input -1 if you want to go to the end
opt_3d = 'save'   # 'save' or 'show' or 'skip'
print_filenames = True
every_n = 1

root_dir = "trained_on_50"  # "trained_on_full", "trained_on_xx"
pcl_type = "7.5"           # "full", "50", "75", etc

classes_list = np.array(["Car"])  # Truck, Car, Pedestrian
###############################################################################3

project_dir = "/home/mrsd2/Documents/vlr-project/"
vrcnn_dir = "/home/mrsd2/Documents/vlr-project/Voxel-R-CNN/"
data_dir = vrcnn_dir + "data/kitti/training/"
img_dir = data_dir + "image_2/"
calib_dir = data_dir + "calib/"
lidar_dir = data_dir + "all_velodynes/" + pcl_dict[pcl_type] + "velodyne/"
label_dir = data_dir + "label_2/"

print(lidar_dir)

pkl_path = project_dir + "/" + root_dir + "/" + pcl_type + "/result.pkl"

class_counter = {}
for c in classes_list:
    class_counter[c] = {'tp':0,'fp':0,'fn':0}

with open(pkl_path, "rb") as file: 
    data = pickle.load(file)

data = preprocess(data)

last_frame = len(data)

if end_frame == -1:
    end_frame = last_frame

total =  int((end_frame - start_frame)/every_n)

print_freq = 100
count = 0

img_creator = ImgCreatorLiDAR()

img_id_list = []
for frame in data[start_frame:end_frame:every_n]:
    if count % print_freq == 0:
        print("Processing frame",count, "/",total)
    count += 1

    img_id = frame['frame_id']
    print("img_id", img_id)
    img = cv2.imread(img_dir + img_id + ".png", -1)
    lidar_path = lidar_dir + img_id + ".bin"
    label_path = label_dir + img_id + ".txt"

    img_id_list.append(img_id)
    
    # POINT CLOUD SETUP ######################################################
    point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

    # remove points that are located behind the camera:
    point_cloud = point_cloud[point_cloud[:, 0] > -2.5, :]
    print(point_cloud.shape)
    
    if print_filenames:
        print(img_id)

    calib = calibread(calib_dir + img_id + ".txt")
    P2 = calib["P2"]
    Tr_velo_to_cam_orig = calib["Tr_velo_to_cam"]
    R0_rect_orig = calib["R0_rect"]
    #
    R0_rect = np.eye(4)
    R0_rect[0:3, 0:3] = R0_rect_orig
    #
    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[0:3, :] = Tr_velo_to_cam_orig
    
    point_cloud_xyz = point_cloud[:, 0:3]
    point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
    point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3] # (point_cloud_xyz_hom has shape (num_points, 4))

    # transform the points into (rectified) camera coordinates:
    point_cloud_xyz_camera_hom = np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T)).T # (point_cloud_xyz_hom.T has shape (4, num_points))
    # normalize:
    point_cloud_xyz_camera = np.zeros((point_cloud_xyz_camera_hom.shape[0], 3))
    point_cloud_xyz_camera[:, 0] = point_cloud_xyz_camera_hom[:, 0]/point_cloud_xyz_camera_hom[:, 3]
    point_cloud_xyz_camera[:, 1] = point_cloud_xyz_camera_hom[:, 1]/point_cloud_xyz_camera_hom[:, 3]
    point_cloud_xyz_camera[:, 2] = point_cloud_xyz_camera_hom[:, 2]/point_cloud_xyz_camera_hom[:, 3]

    pcd = open3d.open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(point_cloud_xyz_camera)
    pcd.paint_uniform_color([0.65, 0.65, 0.65])
    ###########################################################################
    
    
    # BBOX READING #############################################################
    pred_bboxes = []
    pred_bbox_polys = []
    input_2Dbbox_polys = []
    pred_seg_pcds = []
    
    # set confidence score threshold
    scores = frame['score']                        
    valid = scores >= 0.5
    scores = scores[valid]

    class_names = frame['name'][valid]              # Nx1 array, class names of detected boxes
    
    boxes_2d = frame['bbox'][valid,:]               # Nx4 array, 2d bbox coordinates
    
    # boxes_3d = frame['boxes_lidar']               # Nx7 array, 3d box info (doesn't work for visualization)
    # location = boxes_3d[:,:3][valid,:]
    # dims = boxes_3d[:,3:6][valid,:]
    # rotation = boxes_3d[:,6][valid,:]
    
    location = frame['location'][valid,:]            # Nx3, 3d box locations
    dims = frame['dimensions'][valid,:]              # Nx3, 3d box dimensions
    rotation = frame['rotation_y'][valid]            # Nx1, rotations about y (vertical) axis
    
    
    # Calculate the IoU for each class

    # Could calculate the centroid of the gt bbox and then take the box whose centroid is the closest
    # Calculate the box IoU for these two boxes
    # Remove the two boxes from the respective list of boxes
    # Iterate through the amount of ground truth boxes
    # But what if there's an extra gt box? It means there was a false negative, just take note of which gt box was not detected
    # What if there's an extra detected box? IT means there was a false positive, just take note of which gt
    # If there is 0 IoU then maybe the GT box was just not detected bc if there's 0 IoU then it's not really close to the actual object, so dont compare these and say that
    # This gt box was not detected

    
    for c in classes_list:

        gt_boxes_2d, gt_boxes_3d = read_gt_boxes(label_path, c)

        # print(gt_boxes_2d)

        if gt_boxes_2d is not None:
            for i in range(len(gt_boxes_2d)):
                gt_box_2d = gt_boxes_2d[i]
                gt_box_3d = gt_boxes_3d[i]

                gt_2Dbbox_poly = create2Dbbox_poly(gt_box_2d, box_type='gt')
                input_2Dbbox_polys.append(gt_2Dbbox_poly)

                gt_center = gt_box_3d[3:6]
                gt_h,gt_w,gt_l = gt_box_3d[:3]
                gt_r_y = gt_box_3d[6]


                gt_bbox_poly = create3Dbbox_poly(gt_center,gt_h,gt_w,gt_l,gt_r_y,P2,type="gt")
                pred_bbox_polys.append(gt_bbox_poly)

                gt_bbox = create3Dbbox(gt_center,gt_h,gt_w,gt_l,gt_r_y,type="gt")
                pred_bboxes += gt_bbox

        if boxes_2d is not None and boxes_2d.shape[0] != 0:

            pred_box_indices = class_names == c

            pred_boxes_2d = boxes_2d[pred_box_indices,:]
            pred_3d_location = location[pred_box_indices,:]
            pred_3d_dims = dims[pred_box_indices,:]
            pred_3d_rotation = rotation[pred_box_indices]
            pred_scores = scores[pred_box_indices]

            # plot boxes
            for i in range(len(pred_boxes_2d)):
                
                # 2d values
                box_2d = pred_boxes_2d[i]
                
                # 3d values
                pred_center = pred_3d_location[i]
                pred_l,pred_h, pred_w = pred_3d_dims[i]
                pred_r_y = pred_3d_rotation[i]
                
                # Visualization stuff
                input_2Dbbox_poly = create2Dbbox_poly(box_2d,box_type='pred')
                input_2Dbbox_polys.append(input_2Dbbox_poly)

                pred_bbox_poly = create3Dbbox_poly(pred_center, pred_h, pred_w, pred_l, pred_r_y, P2, type="pred")
                pred_bbox_polys.append(pred_bbox_poly)
                
                pred_bbox = create3Dbbox(pred_center, pred_h, pred_w, pred_l, pred_r_y, type="pred")
                pred_bboxes += pred_bbox

            # associate boxes and calculate metrics
            output, tp, fp, fn = associate_boxes_iou_2d(gt_boxes_2d,pred_boxes_2d,iou_thresh = 0.7)

            class_counter[c]['tp'] += tp
            class_counter[c]['fp'] += fp
            class_counter[c]['fn'] += fn

        
    #############################################################################
    
    # FINAL VISUALIZATION 

    import os
    
    save_dir = "./" + root_dir + "/" + pcl_type + "/"
    save_dir_2d = save_dir + "2d_visualizations/"
    save_dir_3d = save_dir + "3d_visualizations/"

    if not os.path.exists(save_dir_2d):
        os.makedirs(save_dir_2d)
    if not os.path.exists(save_dir_3d):
        os.makedirs(save_dir_3d)

    if 'pred_scores' in locals() or 'pred_scores' in globals():
        pass
    else:
        pred_scores = []
    img_with_input_2Dbboxes = draw_2d_polys(img, input_2Dbbox_polys,pred_scores)
    img_with_input_2Dbboxes = cv2.resize(img_with_input_2Dbboxes, (img_width, img_height))

    img_with_pred_bboxes = draw_3d_polys(img, pred_bbox_polys)
    img_with_pred_bboxes = cv2.resize(img_with_pred_bboxes, (img_width, img_height))

    combined_img = np.zeros((2*img_height, img_width, 3), dtype=np.uint8)
    combined_img[0:img_height] = img_with_input_2Dbboxes
    combined_img[img_height:] = img_with_pred_bboxes
    cv2.imwrite(save_dir_2d+img_id+".png", combined_img)

    if opt_3d == 'save':
        img_lidar = img_creator.create_img(pred_bboxes + [pcd])
        cv2.imwrite(save_dir_3d+img_id+".png", img_lidar)
    if opt_3d == 'show':
        draw_geometries_dark_background(pred_seg_pcds + pred_bboxes + [pcd],img_id)
    if opt_3d == 'skip':
        continue

# with open('file_index.txt', 'w') as filehandle:
#     for i, img_id in enumerate(img_id_list):
#         filehandle.write(str(i) + '\t' + img_id + '\n')

print()
print("METRICS")
for c in classes_list:
    print(c)
    

    tp = class_counter[c]['tp']
    fp = class_counter[c]['fp']
    fn = class_counter[c]['fn']
    print("TP",tp)
    print("FP",fp)
    print("FN",fn)

    if (tp+fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    if (tp+fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    print("P",precision)
    print("R",recall)
    print()



