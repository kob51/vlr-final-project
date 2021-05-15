from os import XATTR_SIZE_MAX
import torch
import math
import numpy as np

from .vfe_template import VFETemplate

def create3Dbbox(center, h, w, l, r_y):
    
    Rmat = torch.tensor([[math.cos(r_y), 0, math.sin(r_y)],
                       [0, 1, 0],
                       [-math.sin(r_y), 0, math.cos(r_y)]],
                       dtype=torch.float32)

    # Rmat_90 = torch.tensor([[math.cos(r_y+np.pi/2), 0, math.sin(r_y+np.pi/2)],
    #                       [0, 1, 0],
    #                       [-math.sin(r_y+np.pi/2), 0, math.cos(r_y+np.pi/2)]],
    #                       dtype=torch.float32)

    # Rmat_90_x = torch.tensor([[1, 0, 0],
    #                         [0, math.cos(np.pi/2), math.sin(np.pi/2)],
    #                         [0, -math.sin(np.pi/2), math.cos(np.pi/2)]],
    #                         dtype=torch.float32)

    p0 = center + Rmat @ (torch.tensor([l/2.0, 0, w/2.0], dtype=torch.float32)).flatten()
    p1 = center + Rmat @ (torch.tensor([-l/2.0, 0, w/2.0], dtype=torch.float32)).flatten()
    p2 = center + Rmat @ (torch.tensor([-l/2.0, 0, -w/2.0], dtype=torch.float32)).flatten()
    p3 = center + Rmat @ (torch.tensor([l/2.0, 0, -w/2.0], dtype=torch.float32)).flatten()
    p4 = center + Rmat @ (torch.tensor([l/2.0, -h, w/2.0], dtype=torch.float32)).flatten()
    p5 = center + Rmat @ (torch.tensor([-l/2.0, -h, w/2.0], dtype=torch.float32)).flatten()
    p6 = center + Rmat @ (torch.tensor([-l/2.0, -h, -w/2.0], dtype=torch.float32)).flatten()
    p7 = center + Rmat @ (torch.tensor([l/2.0, -h, -w/2.0], dtype=torch.float32)).flatten()


    # p0 = center + torch.dot(Rmat, torch.tensor([l/2.0, 0, w/2.0], dtype=torch.float32).flatten())
    # p1 = center + torch.dot(Rmat, torch.tensor([-l/2.0, 0, w/2.0], dtype=torch.float32).flatten())
    # p2 = center + torch.dot(Rmat, torch.tensor([-l/2.0, 0, -w/2.0], dtype=torch.float32).flatten())
    # p3 = center + torch.dot(Rmat, torch.tensor([l/2.0, 0, -w/2.0], dtype=torch.float32).flatten())
    # p4 = center + torch.dot(Rmat, torch.tensor([l/2.0, -h, w/2.0], dtype=torch.float32).flatten())
    # p5 = center + torch.dot(Rmat, torch.tensor([-l/2.0, -h, w/2.0], dtype=torch.float32).flatten())
    # p6 = center + torch.dot(Rmat, torch.tensor([-l/2.0, -h, -w/2.0], dtype=torch.float32).flatten())
    # p7 = center + torch.dot(Rmat, torch.tensor([l/2.0, -h, -w/2.0], dtype=torch.float32).flatten())

    return torch.stack((p0,p1,p2,p3,p4,p5,p6,p7)), Rmat



class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, input_batch, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        # print("MEAN VFE")
        query_batch = input_batch[0]
        target_batch = input_batch[1]

        ##########QUERY##########
        print("!!!!!!!!!!!GT BOXES!!!!!!!!!!", query_batch['gt_boxes'].shape)
        box = query_batch['gt_boxes'][:,0,:]
        gt_box = box.squeeze()


        gt_center =  torch.tensor([gt_box[0],gt_box[1],gt_box[2]])

        gt_box_3d_space, Rmat = create3Dbbox(gt_center, gt_box[3],gt_box[4],gt_box[5],gt_box[6])
        # print("!!!!!!!!!!!GT BOXES 3D POINTS!!!!!!!!!!", gt_box_3d_space)
        # print(gt_box_3d_space.shape)

        Rmat_inv = torch.inverse(Rmat).cuda()  # Transform gt box coordinates to an axis aligned frame
        gt_box_3d_trans = torch.transpose(gt_box_3d_space, 0, 1)
        gt_box_3d_trans = Rmat_inv @ gt_box_3d_trans.cuda()


        q_voxel_features, q_voxel_num_points = query_batch['voxels'], query_batch['voxel_num_points']
        # print("!!!!!!!!!!!!PRE VOXELS!!!!!!!!!!!", q_voxel_features)
        q_points_mean = q_voxel_features[:, :, :].sum(dim=1, keepdim=False)
        q_normalizer = torch.clamp_min(q_voxel_num_points.view(-1, 1), min=1.0).type_as(q_voxel_features)
        q_points_mean = q_points_mean / q_normalizer

        q_points_mean_trans = torch.transpose(q_points_mean[:, 0:3], 0, 1)  # Get x,y,z coords and then transpose
        q_points_mean_trans = torch.transpose(Rmat_inv @ q_points_mean_trans.cuda(), 0, 1)

        # print("gt_box_3d_trans", gt_box_3d_trans)
        # print("q_points_mean_trans", q_points_mean_trans)

        xmax = torch.max(gt_box_3d_trans[0, :]).item()
        xmin = torch.min(gt_box_3d_trans[0, :]).item()
        ymax = torch.max(gt_box_3d_trans[1, :]).item()
        ymin = torch.min(gt_box_3d_trans[1, :]).item()
        zmax = torch.max(gt_box_3d_trans[2, :]).item()
        zmin = torch.min(gt_box_3d_trans[2, :]).item()

        # print("MAX/MIN", xmax, xmin, ymax, ymin, zmax, zmin)
        mask1 = q_points_mean_trans[:, 0] <= xmax
        mask2 = q_points_mean_trans[:, 0] >= xmin
        mask3 = q_points_mean_trans[:, 1] <= ymax
        mask4 = q_points_mean_trans[:, 1] >= ymin
        mask5 = q_points_mean_trans[:, 2] <= zmax
        mask6 = q_points_mean_trans[:, 2] >= zmin
        mask = mask1 * mask2 * mask3 * mask4 * mask5 * mask5 * mask6

        # q_points_mean = q_points_mean[mask]
                
        print("!!!!!!!!!!!!QUERY VOXELS!!!!!!!!!!!", q_points_mean.shape)
        # print("!!!!!!!!!!!!QUERY VOXELS!!!!!!!!!!!", q_points_mean[:10,:])
        query_batch['voxel_features'] = q_points_mean.contiguous()
        # print(batch_dict['voxel_features'].shape,"voxel featues")

        ##########TARGET#########
        t_voxel_features, t_voxel_num_points = target_batch['voxels'], target_batch['voxel_num_points']

        t_points_mean = t_voxel_features[:, :, :].sum(dim=1, keepdim=False)
        t_normalizer = torch.clamp_min(t_voxel_num_points.view(-1, 1), min=1.0).type_as(t_voxel_features)
        t_points_mean = t_points_mean / t_normalizer
        # print("!!!!!!!!!!!!TARGET VOXELS!!!!!!!!!!!", t_points_mean.shape)

        target_batch['voxel_features'] = t_points_mean.contiguous()        

        return [query_batch, target_batch]
