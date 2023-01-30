import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.ops import batch_compute_similarity_transform_torch

class MultiViewConsistencyLoss(nn.Module):
    def __init__(self, interval=1):
        super(MultiViewConsistencyLoss, self).__init__()
        self.batch_counter = -1
        # self.interval = interval
        self.interval = 1
        self.coord_loss = nn.L1Loss()

    def forward(self, coord_xyz, view_num, R=None):
        '''
        coord_xyz: shape=(batch, 778+21, 3)
        view_num:
        R: ground-truth rotation to world coordinate system
        '''
        device = coord_xyz.device
        self.batch_counter += 1
        loss_list = []
        avg_fuse = torch.zeros_like(coord_xyz).view(-1, view_num, 799, 3).to(device)
        for view_idx in range(view_num):
            # use ground-truth rotation
            if R is not None:
                batch_relative_R = torch.zeros_like(R).to(device)
                for b in range(batch_relative_R.shape[0]):
                    for j in range(view_num):
                        batch_relative_R[b][j] = R[b][view_idx].matmul(R[b][j].transpose(1, 0))
                gt_R = batch_relative_R.clone().view(-1, 3, 3)

            all_view = coord_xyz
            this_view_single = coord_xyz.view(-1, view_num, 799, 3)[:, view_idx]
            this_view = coord_xyz.view(-1, view_num, 799, 3)[:, view_idx:view_idx+1].repeat(1, view_num, 1, 1).reshape(-1, 799, 3)

            all_view = all_view.permute(0, 2, 1)
            this_view = this_view.permute(0, 2, 1)

            if R is not None:
                aligned_to_this_view, (_, pred_R, _) = batch_compute_similarity_transform_torch(
                    all_view, this_view, gt_R)
            else:
                aligned_to_this_view, (_, pred_R, _) = batch_compute_similarity_transform_torch(
                    all_view, this_view)
                            
            aligned_to_this_view = aligned_to_this_view.permute(0, 2, 1)
            this_view = this_view.permute(0, 2, 1)

            aligned_to_this_view_mean = aligned_to_this_view.reshape(-1, view_num, 799, 3).mean(1)
            avg_fuse[:, view_idx] = aligned_to_this_view_mean

            if self.batch_counter // self.interval % 2 == 0:
                loss_list.append(self.coord_loss(aligned_to_this_view[:, :, :2], this_view.detach()[:, :, :2]))
            else:
                loss_list.append(self.coord_loss(aligned_to_this_view_mean.detach(), this_view_single))
        return loss_list, avg_fuse

    