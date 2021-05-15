import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable

from .anchor_head_template import AnchorHeadTemplate

class match_block(nn.Module):
    def __init__(self, inplanes):
        super(match_block, self).__init__()

        self.sub_sample = False

        self.in_channels = inplanes
        self.inter_channels = None

        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.Q = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.Q[1].weight, 0)
        nn.init.constant_(self.Q[1].bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.concat_project = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
            nn.ReLU()
        )
        
        self.pool = nn.MaxPool2d(2)
        # self.ChannelGate = ChannelGate(self.in_channels)
        # self.globalAvgPool = nn.AdaptiveAvgPool2d(1)


        
    def forward(self, detect, aim):

        detect = self.pool(detect)
        aim = self.pool(aim)

        batch_size, channels, height_a, width_a = aim.shape
        batch_size, channels, height_d, width_d = detect.shape


        #####################################find aim image similar object ####################################################

        d_x = self.g(detect).view(batch_size, self.inter_channels, -1)
        # d_x = self.pool(d_x)
        d_x = d_x.permute(0, 2, 1).contiguous()

        a_x = self.g(aim).view(batch_size, self.inter_channels, -1)
        # a_x = self.pool(a_x)
        a_x = a_x.permute(0, 2, 1).contiguous()

        theta_x = self.theta(aim).view(batch_size, self.inter_channels, -1)
        # theta_x = self.pool(theta_x)
        theta_x = theta_x.permute(0, 2, 1)
                
        phi_x = self.phi(detect).view(batch_size, self.inter_channels, -1)
        # phi_x = self.pool(phi_x)
        # print("theta_x", theta_x.shape)
        # print("phi_x", phi_x.shape)
        f = torch.matmul(theta_x, phi_x)

        N = f.size(-1)
        f_div_C = f / N
        # print("f", f.shape)
        # print("N", N)
        # print("f_div_C", f_div_C.shape)
        # print("d_x", d_x.shape)
        f = f.permute(0, 2, 1).contiguous()
        N = f.size(-1)
        fi_div_C = f / N

        # non_aim = torch.matmul(f_div_C, d_x)
        # non_aim = non_aim.permute(0, 2, 1).contiguous()
        # non_aim = non_aim.view(batch_size, self.inter_channels, height_a, width_a)
        # non_aim = self.W(non_aim)
        # non_aim = non_aim + aim
        # print("a_x.shape", a_x.shape)

        non_det = torch.matmul(fi_div_C, a_x)
        non_det = non_det.permute(0, 2, 1).contiguous()
        non_det = non_det.view(batch_size, self.inter_channels, height_d, width_d)
        # non_det = non_det.view(batch_size, self.inter_channels, 44 , 40)
        non_det = self.Q(non_det)
        non_det = non_det + detect

        ##################################### Response in channel weight ####################################################

        # c_weight = self.ChannelGate(non_aim)
        # act_aim = non_aim * c_weight
        # act_det = non_det * c_weight

        return non_det       #, act_det, act_aim, c_weight

class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        dout_base_model = 256
        self.match_net = match_block(dout_base_model)

        self.upsample = nn.Upsample(scale_factor=2,mode = 'bilinear')
        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):

        q_data_dict = data_dict[0].copy()
        t_data_dict = data_dict[1].copy()

        spatial_features_2d = self.match_net(q_data_dict['spatial_features_2d'],t_data_dict['spatial_features_2d'])

        # print(q_data_dict['spatial_features_2d'].shape)
        # spatial_features_2d = t_data_dict['spatial_features_2d']
        # torch.save(spatial_features_2d,'/home/mrsd2/Desktop/oneshot_spatial_features_2d.pt')
        torch.save(spatial_features_2d,'/home/mrsd2/Desktop/normal_spatial_features_2d.pt')

        spatial_features_2d = self.upsample(spatial_features_2d)
        # torch.save(spatial_features_2d,'/home/mrsd2/Desktop/upsampled_spatial_features_2d.pt')

        # print("spatial features 2d", spatial_features_2d.shape)

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        # print("cls_preds", cls_preds.shape)
        # print("box_preds", box_preds.shape)


        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        # if self.training:
        #     targets_dict = self.assign_targets(
        #         gt_boxes=t_data_dict['gt_boxes']
        #     )
        #     self.forward_ret_dict.update(targets_dict)

        # print("batch size", t_data_dict['batch_size'])
        # print("cls preds", len(cls_preds))
        # print("box preds", len(box_preds))
        # print("dir cls preds", type(dir_cls_preds))


        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size= t_data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            t_data_dict['batch_cls_preds'] = batch_cls_preds
            t_data_dict['batch_box_preds'] = batch_box_preds
            # print(batch_cls_preds.shape,"batch_cls_preds")
            print(batch_box_preds.shape,"batch_box_preds")
            t_data_dict['cls_preds_normalized'] = False

        print("FINAL DICTIONARY",len(t_data_dict))
        return t_data_dict
