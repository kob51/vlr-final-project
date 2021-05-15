#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 16:36:28 2021

@author: mrsd2
"""
import numpy as np

def read_file(filename):
    f = open(filename, "r")
    lines = f.read()
    print(lines)
    return lines.split("\n")

 # Input file name; gets ground truth bboxes for specified classes
def read_gt_boxes(filename, obj_class="Car"): 
    lines = read_file(filename)
    gt_data = []
    for line in lines:
        line = line.split(" ")
        if obj_class in line:
            
            gt_data.append(line)
    gt_boxes = np.array(gt_data)
    if gt_boxes.size == 0:
        return None, None

    gt_boxes_2d = gt_boxes[:, 4:8]
    gt_boxes_2d = gt_boxes_2d.astype(np.float32)

    gt_boxes_3d = gt_boxes[:,8:]
    gt_boxes_3d = gt_boxes_3d.astype(np.float32)
    return  gt_boxes_2d.astype(np.int32), gt_boxes_3d

def iou_2d(box1, box2):
    """
    Calculates Intersection over Union for two bounding boxes (xmin, ymin, xmax, ymax)
    returns IoU vallue
    """
    x1_min, y1_min, x1_max, y1_max = box1[0], box1[1], box1[2], box1[3]
    x2_min, y2_min, x2_max, y2_max = box2[0], box2[1], box2[2], box2[3]
    area1 = (x1_max - x1_min) * (y1_max - y1_min)  # Calculate area of each box
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    xmin = max(x1_min, x2_min)  # Largest x-val of top left corner of bboxes
    ymin = max(y1_min, y2_min)  # Largest y-val of top left corner of bboxes
    xmax = min(x1_max, x2_max)  # Smallest x-val of bottom right corner of bboxes
    ymax = min(y1_max, y2_max)  # Smallest y-val of bottom right corner of bboxes
    intersection = max(xmax - xmin, 0) * max(ymax - ymin, 0)  # If values are negative, there is no intersection
    return intersection / (area1 + area2 - intersection)  # intersection / union

def associate_boxes_centroids(gt_boxes,pred_boxes,iou_thresh = 0.5):
    # N gt boxes
    # M pred boxes
    
    # N x 2
    gt_centroids = np.array([gt_boxes[:,0]/2+gt_boxes[:,2]/2,gt_boxes[:,1]/2 + gt_boxes[:,3]/2]).T
    
    # M x 2
    pred_centroids = np.array([pred_boxes[:,0]/2+pred_boxes[:,2]/2,pred_boxes[:,1]/2 + pred_boxes[:,3]/2]).T
    
    output = []

    false_positives = 0
    false_negatives = 0
    true_positives = 0
    
    while len(gt_centroids) > 0:
        gt_box = gt_boxes[0]
        gt_centroid = gt_centroids[0]
        
        best_dist = 1e5
        best_idx = -1
        
        if len(pred_centroids) > 0:
            for j in range(len(pred_centroids)):
                  dist = np.linalg.norm(gt_centroid-pred_centroids[j])
                  if dist < best_dist:
                     best_dist = dist
                     best_idx = j
            
                
            iou = iou_2d(gt_box,pred_boxes[best_idx])
            if iou >= iou_thresh:
                true_positives += 1
            else:
                false_positives += 1
            output.append((iou,gt_box,pred_boxes[best_idx]))
            
            gt_boxes = np.delete(gt_boxes,0,0)
            gt_centroids = np.delete(gt_centroids,0,0)
            pred_boxes= np.delete(pred_boxes,best_idx,0)
            pred_centroids = np.delete(pred_centroids,best_idx,0)
        
        else:
            iou = 0
            output.append((iou,gt_box,-1))
            
            gt_boxes = np.delete(gt_boxes,0,0)
            gt_centroids = np.delete(gt_centroids,0,0)
    
    return output

def associate_boxes_iou(gt_boxes,pred_boxes,iou_thresh = 0.5):
    # N gt boxes
    # M pred boxes
    
    output = []

    false_positives = 0
    false_negatives = 0
    true_positives = 0
    
    while len(gt_boxes) > 0:
        gt_box = gt_boxes[0]
        
        best_iou = -1
        best_idx = -1
        
        if len(pred_boxes) > 0:
            for j in range(len(pred_boxes)):
                  iou = iou_2d(gt_box,pred_boxes[j])
                  if iou > best_iou:
                     best_iou = iou
                     best_idx = j
            

            if best_iou >= iou_thresh:
                true_positives += 1
            else:
                false_positives += 1
            output.append((best_iou,gt_box,pred_boxes[best_idx]))
            
            gt_boxes = np.delete(gt_boxes,0,0)
            pred_boxes= np.delete(pred_boxes,best_idx,0)
        
        else:
            iou = 0
            output.append((iou,gt_box,-1))
            
            gt_boxes = np.delete(gt_boxes,0,0)
            false_negatives += 1
    
    if len(pred_boxes) > 0:
        false_negatives += len(pred_boxes)
    
    return output, true_positives, false_positives, false_negatives

def nms(bounding_boxes, confidence_score, threshold=0.05):  # Of all the bounding boxes, select the most accurate one
    """  # Select the best bounding box out of the multiple predicted ones
    bounding boxes of shape     Nx4
    confidence scores of shape  N
    threshold: confidence threshold for boxes to be considered
    return: list of bounding boxes and scores
    """

    bounding_boxes = bounding_boxes[confidence_score >= threshold]
    confidence_score = confidence_score[confidence_score >= threshold]
    # Sort remaining bounding boxes and confidence scores in descending order
    sorting_indices = np.argsort(-confidence_score)
    bounding_boxes = bounding_boxes[sorting_indices]
    confidence_score = confidence_score[sorting_indices]
    # Perform non-max suppression
    boxes, scores = [], []  # Lists to store "max" boxes and scores
    while len(bounding_boxes) > 0:  # While there are bounding boxes
        current_box = bounding_boxes[0]  # Compare the box with the top confidence score to all other boxes
        boxes.append(current_box)  # Save top confidence score box
        scores.append(confidence_score[0])  # Save top score
        discard = [0]  # Array to store indices of boxes to discard; discard top box on next iteration since its saved
        for i in range(1, len(bounding_boxes)):  # Iterate through remaining boxes
            current_iou = iou_2d(current_box, bounding_boxes[i])  # Calculate IoU between current box and
            if current_iou >= 0.3:  # If IoU greater than 0.3, discard box and score; 0.3 as described in handout
                discard.append(i)
        bounding_boxes = np.delete(bounding_boxes, discard, axis=0)
        confidence_score = np.delete(confidence_score, discard)
    if len(scores) > 0:
        return np.stack(boxes), np.stack(scores)
    else:
        return boxes, scores
    
    
if __name__ == "__main__":
    # path = "/home/mrsd2/Documents/vlr-project/Voxel-R-CNN/data/kitti/training/label_2/000005.txt"
    
    # obj_class = "Car"
    
    # # box_2d, box_3d = read_gt_boxes(path,obj_class)
    
    # gt_boxes = np.array([[359, 179, 516, 270],
    #                      [444, 163, 559, 247],
    #                      [724, 171, 790, 215],
    #                      [534, 166, 596, 216],
    #                      [709, 162, 765, 205],
    #                      [562, 166, 605, 206]])
    
    # pred_boxes = np.array([[364.01053, 179.39426, 517.32513, 275.57477],
    #                         [727.4303,  170.7321,  789.3416,  214.73586],
    #                         [452.17108, 163.7987,  555.826,   246.75258],
    #                         [540.6479,  171.14978, 596.2053,  217.74545],
    #                         [713.26917, 160.9968,  768.2322,  206.99866]])
    #                         # [562.30023, 169.22482, 605.01105, 208.9798 ]])
    
    
    gt_boxes = np.array([[587, 173, 614, 200],
                         [425, 176, 529, 259],
                         [234, 185, 320, 231]])
    
    pred_boxes = np.array([[422.00824,   175.94032,   528.0782,    259.58627  ],
                         [234.82852,   185.35478,   322.99188,   230.38287  ],
                         [584.539,     173.55173,   614.4193,    198.42737  ],
                         [342.27957,   181.82835,   391.59213,   227.20154  ],
                         [341.75284,   176.0993,    376.50034,   200.06651  ],
                         [  3.3573508, 189.14436,    92.957146,  216.0367   ],
                         [394.5322,    183.41873,   461.57843,   212.18513  ]])
    
    scores = np.array([0.9987581,  0.98727345, 0.8620383,  0.6244919,  0.6045358,  0.5294711,
 0.5042052 ])

    print(pred_boxes.shape,"before nms")
    pred_boxes,scores = nms(pred_boxes,scores)
    print(pred_boxes.shape,"after nms")
    
    output,tp,fp,fn = associate_boxes_iou(gt_boxes,pred_boxes)
    
    print(gt_boxes)
    print(pred_boxes)