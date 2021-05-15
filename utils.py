#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 23:01:38 2021

@author: mrsd2
"""


import math
import json
import numpy as np
import cv2
import open3d

# REFERENCE CODE #########################################
# https://github.com/fregu856/3DOD_thesis
##########################################################

# DATA LOADING HELPER FUNCTIONS ##############################################################
def calibread(file_path):
    out = dict()
    for line in open(file_path, 'r'):
        line = line.strip()
        if line == '' or line[0] == '#':
            continue
        val = line.split(':')
        assert len(val) == 2, 'Wrong file format, only one : per line!'
        key_name = val[0].strip()
        val = np.asarray(val[-1].strip().split(' '), dtype='f8')
        assert len(val) in [12, 9], "Wrong file format, wrong number of numbers!"
        if len(val) == 12:
            out[key_name] = val.reshape(3, 4)
        elif len(val) == 9:
            out[key_name] = val.reshape(3, 3)
    return out

def LabelLoader2D3D(file_id, path, ext, calib_path, calib_ext):
    labels = labelread(path + "/" + file_id + ext)
    calib = calibread(calib_path + "/" + file_id + calib_ext)
    polys = list()
    for bbox in labels:
        poly = dict()

        poly2d = dict()
        poly2d['class'] = bbox['type']
        poly2d['truncated'] = bbox['truncated']
        poly2d['poly'] = np.array([[bbox['bbox']['left'], bbox['bbox']['top']],
                                 [bbox['bbox']['right'], bbox['bbox']['top']],
                                 [bbox['bbox']['right'], bbox['bbox']['bottom']],
                                 [bbox['bbox']['left'], bbox['bbox']['bottom']]],
                                dtype='int32')
        poly["label_2D"] = poly2d

        poly3d = dict()
        poly3d['class'] = bbox['type']
        location = np.asarray([bbox['location']['x'],
                               bbox['location']['y'],
                               bbox['location']['z']], dtype='float32')
        r_y = bbox['rotation_y']
        Rmat = np.asarray([[math.cos(r_y), 0, math.sin(r_y)], [0, 1, 0],
                           [-math.sin(r_y), 0, math.cos(r_y)]],
                          dtype='float32')
        length = bbox['dimensions']['length']
        width = bbox['dimensions']['width']
        height = bbox['dimensions']['height']
        p0 = np.dot(Rmat, np.asarray(
            [length / 2.0, 0, width / 2.0], dtype='float32'))
        p1 = np.dot(Rmat, np.asarray(
            [-length / 2.0, 0, width / 2.0], dtype='float32'))
        p2 = np.dot(Rmat, np.asarray(
            [-length / 2.0, 0, -width / 2.0], dtype='float32'))
        p3 = np.dot(Rmat, np.asarray(
            [length / 2.0, 0, -width / 2.0], dtype='float32'))
        p4 = np.dot(Rmat, np.asarray(
            [length / 2.0, -height, width / 2.0], dtype='float32'))
        p5 = np.dot(Rmat, np.asarray(
            [-length / 2.0, -height, width / 2.0], dtype='float32'))
        p6 = np.dot(Rmat, np.asarray(
            [-length / 2.0, -height, -width / 2.0], dtype='float32'))
        p7 = np.dot(Rmat, np.asarray(
            [length / 2.0, -height, -width / 2.0], dtype='float32'))
        poly3d['points'] = np.array(location + [p0, p1, p2, p3, p4, p5, p6, p7])
        poly3d['lines'] = [[0, 3, 7, 4, 0], [1, 2, 6, 5, 1],
                         [0, 1], [2, 3], [6, 7], [4, 5]]
        poly3d['colors'] = [[255, 0, 0], [0, 0, 255], [
            255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0]]
        poly3d['P0_mat'] = calib['P2']
        poly3d['center'] = location
        poly3d['l'] = length
        poly3d['w'] = width
        poly3d['h'] = height
        poly3d['r_y'] = r_y
        poly["label_3D"] = poly3d

        polys.append(poly)
    return polys

def LabelLoader2D3D_sequence(img_id, img_id_float, label_path, calib_path):
    labels = labelread_sequence(label_path)

    img_id_labels = []
    for label in labels:
        if label["frame"] == img_id_float:
            img_id_labels.append(label)

    calib = calibread(calib_path)
    polys = list()
    for bbox in img_id_labels:
        poly = dict()

        poly2d = dict()
        poly2d['class'] = bbox['type']
        poly2d['truncated'] = bbox['truncated']
        poly2d['occluded'] = bbox['occluded']
        poly2d['poly'] = np.array([[bbox['bbox']['left'], bbox['bbox']['top']],
                                 [bbox['bbox']['right'], bbox['bbox']['top']],
                                 [bbox['bbox']['right'], bbox['bbox']['bottom']],
                                 [bbox['bbox']['left'], bbox['bbox']['bottom']]],
                                dtype='int32')
        poly["label_2D"] = poly2d

        poly3d = dict()
        poly3d['class'] = bbox['type']
        location = np.asarray([bbox['location']['x'],
                               bbox['location']['y'],
                               bbox['location']['z']], dtype='float32')
        r_y = bbox['rotation_y']
        Rmat = np.asarray([[math.cos(r_y), 0, math.sin(r_y)], [0, 1, 0],
                           [-math.sin(r_y), 0, math.cos(r_y)]],
                          dtype='float32')
        length = bbox['dimensions']['length']
        width = bbox['dimensions']['width']
        height = bbox['dimensions']['height']
        p0 = np.dot(Rmat, np.asarray(
            [length / 2.0, 0, width / 2.0], dtype='float32'))
        p1 = np.dot(Rmat, np.asarray(
            [-length / 2.0, 0, width / 2.0], dtype='float32'))
        p2 = np.dot(Rmat, np.asarray(
            [-length / 2.0, 0, -width / 2.0], dtype='float32'))
        p3 = np.dot(Rmat, np.asarray(
            [length / 2.0, 0, -width / 2.0], dtype='float32'))
        p4 = np.dot(Rmat, np.asarray(
            [length / 2.0, -height, width / 2.0], dtype='float32'))
        p5 = np.dot(Rmat, np.asarray(
            [-length / 2.0, -height, width / 2.0], dtype='float32'))
        p6 = np.dot(Rmat, np.asarray(
            [-length / 2.0, -height, -width / 2.0], dtype='float32'))
        p7 = np.dot(Rmat, np.asarray(
            [length / 2.0, -height, -width / 2.0], dtype='float32'))
        poly3d['points'] = np.array(location + [p0, p1, p2, p3, p4, p5, p6, p7])
        poly3d['lines'] = [[0, 3, 7, 4, 0], [1, 2, 6, 5, 1],
                         [0, 1], [2, 3], [6, 7], [4, 5]]
        poly3d['colors'] = [[255, 0, 0], [0, 0, 255], [
            255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0]]
        poly3d['P0_mat'] = calib['P2']
        poly3d['center'] = location
        poly3d['l'] = length
        poly3d['w'] = width
        poly3d['h'] = height
        poly3d['r_y'] = r_y
        poly["label_3D"] = poly3d

        polys.append(poly)
    return polys

def labelread(file_path):
    bbox = ('bbox', ['left', 'top', 'right', 'bottom'])
    dimensions = ('dimensions', ['height', 'width', 'length'])
    location = ('location', ['x', 'y', 'z'])
    keys = ['type', 'truncated', 'occluded', 'alpha', bbox,
            dimensions, location, 'rotation_y', 'score']
    labels = list()
    for line in open(file_path, 'r'):
        vals = line.split()
        l, _ = vals_to_dict(vals, keys)
        labels.append(l)
    return labels

def labelread_sequence(file_path):
    bbox = ('bbox', ['left', 'top', 'right', 'bottom'])
    dimensions = ('dimensions', ['height', 'width', 'length'])
    location = ('location', ['x', 'y', 'z'])
    keys = ['frame', 'track_id', 'type', 'truncated', 'occluded', 'alpha', bbox,
            dimensions, location, 'rotation_y', 'score']
    labels = list()
    for line in open(file_path, 'r'):
        vals = line.split()
        l, _ = vals_to_dict(vals, keys)
        labels.append(l)
    return labels

def vals_to_dict(vals, keys, vals_n=0):
    out = dict()
    for key in keys:
        if isinstance(key, str):
            try:
                val = float(vals[vals_n])
            except:
                val = vals[vals_n]
            data = val
            key_name = key
            vals_n += 1
        else:
            data, vals_n = vals_to_dict(vals, key[1], vals_n)
            key_name = key[0]
        out[key_name] = data
        if vals_n >= len(vals):
            break
    return out, vals_n

# Read txt file; returns list where each entry is a line in the txt file
def read_file(filename):
    f = open(filename, "r")
    lines = f.read()
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

def centroid(boxes):
    # Calculate the centroids of boxes; input is list of boxes
    centroids_list = []
    for box in boxes:
        xc = (box[0] + box[2]) / 2
        yc = (box[1] + box[3]) / 2
        centroids_list.append([xc, yc])
    return np.stack(centroids_list)
##############################################################################################

# VISUALIZATION HELPER FUNCTIONS #############################################################
def create3Dbbox(center, h, w, l, r_y, type="pred"):
    if type == "pred":
        color = [1, 0, 0] # (normalized RGB)
        front_color = [1, 0, 0] # (normalized RGB)
    else: # (if type == "gt":)
        color = [0, 1, 0] # (normalized RGB)
        front_color = [0, 1, 0] # (normalized RGB)

    Rmat = np.asarray([[math.cos(r_y), 0, math.sin(r_y)],
                       [0, 1, 0],
                       [-math.sin(r_y), 0, math.cos(r_y)]],
                       dtype='float32')

    Rmat_90 = np.asarray([[math.cos(r_y+np.pi/2), 0, math.sin(r_y+np.pi/2)],
                          [0, 1, 0],
                          [-math.sin(r_y+np.pi/2), 0, math.cos(r_y+np.pi/2)]],
                          dtype='float32')

    Rmat_90_x = np.asarray([[1, 0, 0],
                            [0, math.cos(np.pi/2), math.sin(np.pi/2)],
                            [0, -math.sin(np.pi/2), math.cos(np.pi/2)]],
                            dtype='float32')

    p0 = center + np.dot(Rmat, np.asarray([l/2.0, 0, w/2.0], dtype='float32').flatten())
    p1 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, w/2.0], dtype='float32').flatten())
    p2 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, -w/2.0], dtype='float32').flatten())
    p3 = center + np.dot(Rmat, np.asarray([l/2.0, 0, -w/2.0], dtype='float32').flatten())
    p4 = center + np.dot(Rmat, np.asarray([l/2.0, -h, w/2.0], dtype='float32').flatten())
    p5 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, w/2.0], dtype='float32').flatten())
    p6 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, -w/2.0], dtype='float32').flatten())
    p7 = center + np.dot(Rmat, np.asarray([l/2.0, -h, -w/2.0], dtype='float32').flatten())

    p0_3 = center + np.dot(Rmat, np.asarray([l/2.0, 0, 0], dtype='float32').flatten())
    p1_2 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, 0], dtype='float32').flatten())
    p4_7 = center + np.dot(Rmat, np.asarray([l/2.0, -h, 0], dtype='float32').flatten())
    p5_6 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, 0], dtype='float32').flatten())
    p0_1 = center + np.dot(Rmat, np.asarray([0, 0, w/2.0], dtype='float32').flatten())
    p3_2 = center + np.dot(Rmat, np.asarray([0, 0, -w/2.0], dtype='float32').flatten())
    p4_5 = center + np.dot(Rmat, np.asarray([0, -h, w/2.0], dtype='float32').flatten())
    p7_6 = center + np.dot(Rmat, np.asarray([0, -h, -w/2.0], dtype='float32').flatten())
    p0_4 = center + np.dot(Rmat, np.asarray([l/2.0, -h/2.0, w/2.0], dtype='float32').flatten())
    p3_7 = center + np.dot(Rmat, np.asarray([l/2.0, -h/2.0, -w/2.0], dtype='float32').flatten())
    p1_5 = center + np.dot(Rmat, np.asarray([-l/2.0, -h/2.0, w/2.0], dtype='float32').flatten())
    p2_6 = center + np.dot(Rmat, np.asarray([-l/2.0, -h/2.0, -w/2.0], dtype='float32').flatten())
    p0_1_3_2 = center

    length_0_3 = np.linalg.norm(p0 - p3)
    cylinder_0_3 = open3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_0_3)
    cylinder_0_3.compute_vertex_normals()
    transform_0_3 = np.eye(4)
    transform_0_3[0:3, 0:3] = Rmat
    transform_0_3[0:3, 3] = p0_3
    cylinder_0_3.transform(transform_0_3)
    cylinder_0_3.paint_uniform_color(front_color)

    length_1_2 = np.linalg.norm(p1 - p2)
    cylinder_1_2 = open3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_1_2)
    cylinder_1_2.compute_vertex_normals()
    transform_1_2 = np.eye(4)
    transform_1_2[0:3, 0:3] = Rmat
    transform_1_2[0:3, 3] = p1_2
    cylinder_1_2.transform(transform_1_2)
    cylinder_1_2.paint_uniform_color(color)

    length_4_7 = np.linalg.norm(p4 - p7)
    cylinder_4_7 = open3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_4_7)
    cylinder_4_7.compute_vertex_normals()
    transform_4_7 = np.eye(4)
    transform_4_7[0:3, 0:3] = Rmat
    transform_4_7[0:3, 3] = p4_7
    cylinder_4_7.transform(transform_4_7)
    cylinder_4_7.paint_uniform_color(front_color)

    length_5_6 = np.linalg.norm(p5 - p6)
    cylinder_5_6 = open3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_5_6)
    cylinder_5_6.compute_vertex_normals()
    transform_5_6 = np.eye(4)
    transform_5_6[0:3, 0:3] = Rmat
    transform_5_6[0:3, 3] = p5_6
    cylinder_5_6.transform(transform_5_6)
    cylinder_5_6.paint_uniform_color(color)

    # #

    length_0_1 = np.linalg.norm(p0 - p1)
    cylinder_0_1 = open3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_0_1)
    cylinder_0_1.compute_vertex_normals()
    transform_0_1 = np.eye(4)
    transform_0_1[0:3, 0:3] = Rmat_90
    transform_0_1[0:3, 3] = p0_1
    cylinder_0_1.transform(transform_0_1)
    cylinder_0_1.paint_uniform_color(color)

    length_3_2 = np.linalg.norm(p3 - p2)
    cylinder_3_2 = open3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_3_2)
    cylinder_3_2.compute_vertex_normals()
    transform_3_2 = np.eye(4)
    transform_3_2[0:3, 0:3] = Rmat_90
    transform_3_2[0:3, 3] = p3_2
    cylinder_3_2.transform(transform_3_2)
    cylinder_3_2.paint_uniform_color(color)

    length_4_5 = np.linalg.norm(p4 - p5)
    cylinder_4_5 = open3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_4_5)
    cylinder_4_5.compute_vertex_normals()
    transform_4_5 = np.eye(4)
    transform_4_5[0:3, 0:3] = Rmat_90
    transform_4_5[0:3, 3] = p4_5
    cylinder_4_5.transform(transform_4_5)
    cylinder_4_5.paint_uniform_color(color)

    length_7_6 = np.linalg.norm(p7 - p6)
    cylinder_7_6 = open3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_7_6)
    cylinder_7_6.compute_vertex_normals()
    transform_7_6 = np.eye(4)
    transform_7_6[0:3, 0:3] = Rmat_90
    transform_7_6[0:3, 3] = p7_6
    cylinder_7_6.transform(transform_7_6)
    cylinder_7_6.paint_uniform_color(color)

    # #

    length_0_4 = np.linalg.norm(p0 - p4)
    cylinder_0_4 = open3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_0_4)
    cylinder_0_4.compute_vertex_normals()
    transform_0_4 = np.eye(4)
    transform_0_4[0:3, 0:3] = np.dot(Rmat, Rmat_90_x)
    transform_0_4[0:3, 3] = p0_4
    cylinder_0_4.transform(transform_0_4)
    cylinder_0_4.paint_uniform_color(front_color)

    length_3_7 = np.linalg.norm(p3 - p7)
    cylinder_3_7 = open3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_3_7)
    cylinder_3_7.compute_vertex_normals()
    transform_3_7 = np.eye(4)
    transform_3_7[0:3, 0:3] = np.dot(Rmat, Rmat_90_x)
    transform_3_7[0:3, 3] = p3_7
    cylinder_3_7.transform(transform_3_7)
    cylinder_3_7.paint_uniform_color(front_color)

    length_1_5 = np.linalg.norm(p1 - p5)
    cylinder_1_5 = open3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_1_5)
    cylinder_1_5.compute_vertex_normals()
    transform_1_5 = np.eye(4)
    transform_1_5[0:3, 0:3] = np.dot(Rmat, Rmat_90_x)
    transform_1_5[0:3, 3] = p1_5
    cylinder_1_5.transform(transform_1_5)
    cylinder_1_5.paint_uniform_color(color)

    length_2_6 = np.linalg.norm(p2 - p6)
    cylinder_2_6 = open3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_2_6)
    cylinder_2_6.compute_vertex_normals()
    transform_2_6 = np.eye(4)
    transform_2_6[0:3, 0:3] = np.dot(Rmat, Rmat_90_x)
    transform_2_6[0:3, 3] = p2_6
    cylinder_2_6.transform(transform_2_6)
    cylinder_2_6.paint_uniform_color(color)

    # #

    length_0_1_3_2 = np.linalg.norm(p0_1 - p3_2)
    cylinder_0_1_3_2 = open3d.geometry.TriangleMesh.create_cylinder(radius=0.025, height=length_0_1_3_2)
    cylinder_0_1_3_2.compute_vertex_normals()
    transform_0_1_3_2 = np.eye(4)
    transform_0_1_3_2[0:3, 0:3] = Rmat
    transform_0_1_3_2[0:3, 3] = p0_1_3_2
    cylinder_0_1_3_2.transform(transform_0_1_3_2)
    cylinder_0_1_3_2.paint_uniform_color(color)

    return [cylinder_0_3, cylinder_1_2, cylinder_4_7, cylinder_5_6, cylinder_0_1, cylinder_3_2, cylinder_4_5, cylinder_7_6, cylinder_0_4, cylinder_3_7, cylinder_1_5, cylinder_2_6]

def create3Dbbox_poly(center, h, w, l, r_y, P2_mat, type="pred"):
    if type == "pred":
        color = [0, 0, 255] # (BGR)
        front_color = [0, 0, 255] # (BGR)
    else: # (if type == "gt":)
        color = [0, 255, 0] # (BGR)
        front_color = [0, 255, 0] # (BGR)

    poly = {}

    Rmat = np.asarray([[math.cos(r_y), 0, math.sin(r_y)],
                       [0, 1, 0],
                       [-math.sin(r_y), 0, math.cos(r_y)]],
                       dtype='float32')

    p0 = center + np.dot(Rmat, np.asarray([l/2.0, 0, w/2.0], dtype='float32').flatten())
    p1 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, w/2.0], dtype='float32').flatten())
    p2 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, -w/2.0], dtype='float32').flatten())
    p3 = center + np.dot(Rmat, np.asarray([l/2.0, 0, -w/2.0], dtype='float32').flatten())
    p4 = center + np.dot(Rmat, np.asarray([l/2.0, -h, w/2.0], dtype='float32').flatten())
    p5 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, w/2.0], dtype='float32').flatten())
    p6 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, -w/2.0], dtype='float32').flatten())
    p7 = center + np.dot(Rmat, np.asarray([l/2.0, -h, -w/2.0], dtype='float32').flatten())

    poly['points'] = np.array([p0, p1, p2, p3, p4, p5, p6, p7])
    poly['lines'] = [[0, 3, 7, 4, 0], [1, 2, 6, 5, 1], [0, 1], [2, 3], [6, 7], [4, 5]] # (0 -> 3 -> 7 -> 4 -> 0, 1 -> 2 -> 6 -> 5 -> 1, etc.)
    poly['colors'] = [front_color, color, color, color, color, color]
    poly['P0_mat'] = P2_mat

    return poly

def create2Dbbox_poly(bbox2D,box_type='pred'):
    u_min = bbox2D[0] # (left)
    v_min = bbox2D[1] # (top)
    u_max = bbox2D[2] # (right)
    v_max = bbox2D[3] # (bottom)

    poly = {}
    poly['poly'] = np.array([[u_min, v_min], [u_max, v_min],
                             [u_max, v_max], [u_min, v_max]], dtype='int32')


    if box_type == 'pred':
        color = np.array((0,0,255),dtype='float64')
    if box_type == 'gt':
        color = np.array((0,255,0),dtype='float64')

    poly['color'] = color

    return poly

def draw_2d_polys(img, polys, scores):
    img = np.copy(img)
    i=0
    for poly in polys:

        if 'color' in poly:
            bg = poly['color']
        else:
            bg = np.array([0, 255, 0], dtype='float64')

        if (bg == np.array([0, 0, 255], dtype='float64')).all():
            font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(img,"{:.2f}".format(scores[i]),tuple(np.int32([poly['poly']])[0,0]), font, 0.5, bg, 1, cv2.LINE_AA)
            i+=1

        cv2.polylines(img, np.int32([poly['poly']]), True, bg, lineType=cv2.LINE_AA, thickness=2)    

    return img

def draw_3d_polys(img, polys):
    img = np.copy(img)
    for poly in polys:
        for n, line in enumerate(poly['lines']):
            if 'colors' in poly:
                bg = poly['colors'][n]
            else:
                bg = np.array([255, 0, 0], dtype='float64')

            p3d = np.vstack((poly['points'][line].T, np.ones((1, poly['points'][line].shape[0]))))
            p2d = np.dot(poly['P0_mat'], p3d)

            for m, p in enumerate(p2d[2, :]):
                p2d[:, m] = p2d[:, m]/p

            cv2.polylines(img, np.int32([p2d[:2, :].T]), False, bg, lineType=cv2.LINE_AA, thickness=2)

    return img

def draw_geometries_dark_background(geometries,img_id,show_3d=True):

    # load json file with saved viewpoint
    with open('render_setting.json') as f:
      data = json.load(f)
    traj = data['trajectory'][0]

    mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=3)
    geometries.append(mesh_frame)
    vis = open3d.visualization.Visualizer()
    
    vis.create_window(visible = show_3d)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    
    for geometry in geometries:
        vis.add_geometry(geometry)
    
    # set custom viewpoint from json file
    ctr = vis.get_view_control()
    ctr.change_field_of_view(traj['field_of_view'])
    ctr.set_lookat(traj['lookat'])
    ctr.set_up(traj['up'])
    ctr.set_zoom(traj['zoom'])
    ctr.set_front(traj['front'])
    
    
    vis.run()
    # vis.update_renderer()
    
    img = vis.capture_screen_float_buffer()
    img = 255*np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.uint8)
    cv2.imwrite("./3d_visualizations/" + img_id +".png",img)
    
    
    vis.destroy_window()
    
class ImgCreatorLiDAR:
    def __init__(self):
        with open('render_setting.json') as f:
            data = json.load(f)
        self.traj = data['trajectory'][0]

        self.counter = 0
        # self.trajectory = read_pinhole_camera_trajectory("/home/fregu856/3DOD_thesis/visualization/camera_trajectory.json") # NOTE! you'll have to adapt this for your file structure

    def move_forward(self, vis):
        # this function is called within the Visualizer::run() loop.
        # the run loop calls the function, then re-renders the image.

        if self.counter < 2: # (the counter is for making sure the camera view has been changed before the img is captured)
            # set the camera view:
            ctr = vis.get_view_control()

            ctr = vis.get_view_control()
            ctr.change_field_of_view(self.traj['field_of_view'])
            ctr.set_lookat(self.traj['lookat'])
            ctr.set_up(self.traj['up'])
            ctr.set_zoom(self.traj['zoom'])
            ctr.set_front(self.traj['front'])
            # ctr.convert_from_pinhole_camera_parameters(self.trajectory.intrinsic, self.trajectory.extrinsic[0])

            self.counter += 1
        else:
            # capture an image:
            img = vis.capture_screen_float_buffer()
            img = 255*np.asarray(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.uint8)
            self.lidar_img = img

            # close the window:
            vis.destroy_window()

            self.counter = 0

        return False

    def create_img(self, geometries):
        vis = open3d.visualization.Visualizer()
        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        for geometry in geometries:
            vis.add_geometry(geometry)
        vis.register_animation_callback(self.move_forward)
        vis.run()

        return self.lidar_img


###############################################################################################

# class ImgCreatorLiDAR:
#     def __init__(self):
#         self.counter = 0
#         self.trajectory = "/home/mrsd2/Documents/vlr-project/camera.json"

#     def move_forward(self, vis):
#         # this function is called within the Visualizer::run() loop.
#         # the run loop calls the function, then re-renders the image.

#         if self.counter < 2: # (the counter is for making sure the camera view has been changed before the img is captured)
#             # set the camera view:
#             ctr = vis.get_view_control()
#             ctr.convert_from_pinhole_camera_parameters(self.trajectory.intrinsic, self.trajectory.extrinsic[0])

#             self.counter += 1
#         else:
#             # capture an image:
#             img = vis.capture_screen_float_buffer()
#             img = 255*np.asarray(img)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = img.astype(np.uint8)
#             self.lidar_img = img

#             # close the window:
#             vis.destroy_window()

#             self.counter = 0

#         return False

#     def create_img(self, geometries):
#         vis = open3d.visualization.Visualizer()
#         vis.create_window()
#         opt = vis.get_render_option()
#         opt.background_color = np.asarray([0, 0, 0])
#         for geometry in geometries:
#             vis.add_geometry(geometry)
#         # vis.run()
#         img = vis.capture_screen_float_buffer()
#         img = 255*np.asarray(img)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = img.astype(np.uint8)
#         self.lidar_img = img 
       
        

#         return self.lidar_img

# END 3rd PARTY CODE ##############################################################





# OUR CODE ##################################################################

def preprocess(dict_data):
    for item in dict_data:
        item['frame_id'] = str(item['frame_id'])
    
    return dict_data

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
            if current_iou >= 0.1:  # If IoU greater than 0.3, discard box and score; 0.3 as described in handout
                discard.append(i)
        bounding_boxes = np.delete(bounding_boxes, discard, axis=0)
        confidence_score = np.delete(confidence_score, discard)
    if len(scores) > 0:
        return np.stack(boxes), np.stack(scores)
    else:
        return boxes, scores

def associate_boxes_iou_2d(gt_boxes,pred_boxes,iou_thresh = 0.7):
    output = []

    false_positives = 0
    false_negatives = 0
    true_positives = 0
    
    if gt_boxes is None:
        if pred_boxes is None:
            return None, 0, 0, 0
        else:
            return None, 0, len(pred_boxes), 0
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