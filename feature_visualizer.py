#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 17:17:10 2021

@author: mrsd2
"""

import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path1 = "/home/mrsd2/Desktop/oneshot_spatial_features_2d.pt"
    path2 = "/home/mrsd2/Desktop/upsampled_spatial_features_2d.pt"
    path3 = "/home/mrsd2/Desktop/normal_spatial_features_2d.pt"
    
    one = torch.load(path1).cpu().numpy()
    two = torch.load(path2).cpu().numpy()
    three = torch.load(path3).cpu().numpy()

    
    print(one.shape)
    
    
    for i in range(256):
        img = three[0,i,:,:]
        
        plt.title(str(i))
        plt.imshow(img)
        plt.show()