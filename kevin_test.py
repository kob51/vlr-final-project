# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle

# import string

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def preprocess(dict_data):
    for item in dict_data:
        item['frame_id'] = str(item['frame_id'])
    return dict_data

if __name__ == "__main__":
    
    places = ['Berlin', 'Cape Town', 'Sydney', 'Moscow']

    with open('listfile.txt', 'w') as filehandle:
        for i, img_id in enumerate(places):
            filehandle.write(str(i) + '\t' + img_id + '\n')
            
    # fuck
    
    plot = False
    
    classes = set()
    
    # pkl_path = "/home/mrsd2/Documents/vlr-project/Voxel-R-CNN/output/voxel_rcnn/voxel_rcnn_car/default/eval/eval_with_train/epoch_80/val/result.pkl"
    pkl_path = "/home/mrsd2/Documents/vlr-project/oneshot_voxelrcnn/output/voxel_rcnn/voxel_rcnn_pedestrian/default/eval/epoch_80/val/default/result.pkl"
    
    image_path = "/home/mrsd2/Documents/vlr-project/Voxel-R-CNN/data/kitti/training/image_2/"
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    desired_frame = '000143'
    
    data = preprocess(data)
    
    filenames = [x['frame_id'] for x in data]
    
    data_dict = data[0]
    
    out = None
    for x in data:
        if x['frame_id'] == desired_frame:
            out = x
            break
    
    
    
    testpath = "/home/mrsd2/Documents/vlr-project/Voxel-R-CNN/data/kitti"
    
    
        
    #     if plot:
    #         fig, ax = plt.subplots()
    #         img = plt.imread(image_path + item['frame_id']+".png")
    #         ax.imshow(img)
    #         for i in range(len(item['bbox'])):
                
    #             if item['score'][i] < 0.5:
    #                 continue
                
    #             x1,y1,x2,y2 = item['bbox'][i]
                
    #             rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
    #             ax.add_patch(rect)
    #     plt.show()
        