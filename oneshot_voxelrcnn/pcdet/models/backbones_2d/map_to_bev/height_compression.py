import torch.nn as nn
import torch


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, input_batch):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        # print("HEIGHT COMPRESSION")
        query_batch = input_batch[0]
        target_batch = input_batch[1]
        #######QUERY#########
        q_encoded_spconv_tensor = query_batch['encoded_spconv_tensor']
        q_spatial_features = q_encoded_spconv_tensor.dense()
        N, C, D, H, W = q_spatial_features.shape
        q_spatial_features = q_spatial_features.view(N, C * D, H, W)
        query_batch['spatial_features'] = q_spatial_features
        # print(spatial_features.shape,"spatial features")
        # torch.save(q_spatial_features,'/home/mrsd2/Desktop/spatial_features_hc.pt')
        query_batch['spatial_features_stride'] = query_batch['encoded_spconv_tensor_stride']
        
        ######TARGET########
        t_encoded_spconv_tensor = target_batch['encoded_spconv_tensor']
        t_spatial_features = t_encoded_spconv_tensor.dense()
        N, C, D, H, W = t_spatial_features.shape
        t_spatial_features = t_spatial_features.view(N, C * D, H, W)
        target_batch['spatial_features'] = t_spatial_features
        # print(spatial_features.shape,"spatial features")
        # torch.save(t_spatial_features,'/home/mrsd2/Desktop/spatial_features_hc.pt')
        target_batch['spatial_features_stride'] = target_batch['encoded_spconv_tensor_stride']
        
        return [query_batch, target_batch]

