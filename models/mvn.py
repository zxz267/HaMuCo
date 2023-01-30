import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm
from torchvision import models
from collections import defaultdict

from utils.mano import MANO
from config import cfg
from models.module import GraphRegression, PixelFeatureSampler, SAIGB, SelfAttn, ManoDecoder
from models.loss import MultiViewConsistencyLoss
from models.base import BaseModel
from models.model import MODEL_REGISTRY

from utils.ops import batch_compute_similarity_transform_torch


# cross-view interaction
class CrossViewInteraction(nn.Module):
    def __init__(self, view_num, joint_num, in_dim, out_dim):
        super(CrossViewInteraction, self).__init__()
        self.view_num = view_num
        self.location_embedding_dim = 64
        # resnet
        self.feat_dim = cfg.feature_size * cfg.num_FMs + in_dim // 8 + in_dim // 4 + in_dim // 2 + self.location_embedding_dim
        self.pose_mapping = nn.Sequential(
            nn.Linear(joint_num * 3 + 6 * 15, joint_num * self.location_embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(joint_num * self.location_embedding_dim, joint_num * self.location_embedding_dim),
        )
        self.saigb_pose = SAIGB(in_dim, cfg.num_FMs, cfg.feature_size, cfg.num_joint)
        # two-branch transformer
        self.gcn_0 = GraphRegression(joint_num, self.feat_dim, self.feat_dim, last=False)
        self.gcn_1 = GraphRegression(joint_num, self.feat_dim, self.feat_dim, last=False)
        self.att_0 = SelfAttn(self.feat_dim, self.feat_dim, dropout=0)
        self.att_1 = SelfAttn(self.feat_dim, self.feat_dim, dropout=0)
        # mano parameters regression
        self.fuse = GraphRegression(joint_num, self.feat_dim, out_dim, last=False)
        self.fc = nn.Sequential(
            nn.Linear(out_dim * joint_num, out_dim * joint_num),
            nn.LeakyReLU(0.1),
            nn.Linear(out_dim * joint_num, 16 * 6 + 10 + 3),
        )
        self.decoder = ManoDecoder()

    def forward(self, feat_pose, jaf, joint_uvd, prev_mano_params, view_num=None):
        if view_num is not None:
            self.view_num = view_num

        # multi-view graph building
        feat_pose = self.saigb_pose(feat_pose)
        batch, joint_num, feat_dim = feat_pose.shape
        localization_feat = self.pose_mapping(torch.cat((joint_uvd.reshape(-1, joint_num * 3), prev_mano_params[:, 6:6*16].reshape(-1, 6*15)), dim=1))
        mv_feat = torch.cat((
            feat_pose.reshape(-1, joint_num, feat_dim), 
            jaf.reshape(-1, joint_num, jaf.shape[-1]), 
            localization_feat.view(-1, joint_num, 64)), dim=2)

        new_feat = mv_feat.reshape(-1, self.view_num * joint_num, self.feat_dim)

        if cfg.random_mask and self.training:
            use_view_num = random.randint(1, cfg.num_view)
            use_view_set = random.sample([i for i in range(cfg.num_view)], use_view_num)
            use_view_set.sort()
        else:
            use_view_set = [i for i in range(cfg.num_view)]

        # view-shared feature - max
        canonical_feat = self.gcn_0(new_feat.view(-1, joint_num, self.feat_dim)) 
        canonical_feat = canonical_feat.view(-1, self.view_num, joint_num, self.feat_dim)[:, use_view_set].max(1)[0].repeat(1, self.view_num, 1) 
        # attention feature
        att_aug_feat = self.att_0(new_feat, mask=cfg.random_mask)
        # two-branch resisual
        new_feat = att_aug_feat + canonical_feat

        if cfg.random_mask and self.training:
            use_view_num = random.randint(1, cfg.num_view)
            use_view_set = random.sample([i for i in range(cfg.num_view)], use_view_num)
            use_view_set.sort()
        else:
            use_view_set = [i for i in range(cfg.num_view)]

        # view-shared feature - max
        canonical_feat = self.gcn_1(new_feat.view(-1, joint_num, self.feat_dim)) 
        canonical_feat = canonical_feat.view(-1, self.view_num, joint_num, self.feat_dim)[:, use_view_set].max(1)[0].repeat(1, self.view_num, 1) 

        att_aug_feat = self.att_1(new_feat, mask=cfg.random_mask)

        # two-branch resisual
        new_feat = att_aug_feat + canonical_feat
        # new_feat = new_feat + canonical_feat

        # mano parameters regression
        new_feat = self.fuse(new_feat.reshape(-1, joint_num, new_feat.shape[-1]))
        new_feat = new_feat.view(-1, joint_num * new_feat.shape[-1])
        mano_params = self.fc(new_feat) #+ prev_mano_params.detach()

        # mano decoder
        cam = mano_params[:, 16*6+10:]
        cam = torch.cat((F.relu(cam[:, 0:1]), cam[:, 1:]), dim=1).view(batch, 3)
        pose = mano_params[:, :16*6]
        shape = prev_mano_params[:, 16*6:16*6+10].detach()
        coord_xyz, coord_uvd, pose, shape, cam = self.decoder(pose, shape, cam)
        return coord_xyz, coord_uvd, pose, shape, cam

class SimpleHead(nn.Module):
    def __init__(self, channel):
        super(SimpleHead, self).__init__()
        self.layers_pose = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channel, channel),
            nn.LeakyReLU(0.1),
            nn.Linear(channel, 6 * 16)
        )       
        self.layers_shape = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channel, channel),
            nn.LeakyReLU(0.1),
            nn.Linear(channel, 10)
        )       
        self.layers_cam = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channel, channel),
            nn.LeakyReLU(0.1),
            nn.Linear(channel, 3)
        )       
        self.decoder = ManoDecoder()
    
    def forward(self, feat_high):
        pose = self.layers_pose(feat_high)
        shape = self.layers_shape(feat_high)
        cam = self.layers_cam(feat_high)
        batch = pose.shape[0]
        # positive scale
        cam = torch.cat((F.relu(cam[:, 0:1]), cam[:, 1:]), dim=1).view(batch, 3)
        mano_params = torch.cat((pose, shape, cam), dim=1)
        # mano decoder
        coord_xyz, coord_uvd, pose, shape, cam = self.decoder(pose, shape, cam)
        return coord_xyz, coord_uvd, pose, shape, mano_params

class SingleViewNet(nn.Module):
    def __init__(self):
        super(SingleViewNet, self).__init__()
        backbone = models.__dict__[cfg.backbone](pretrained=True)
        self.channel = backbone.fc.in_features
        self.extract_mid = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu,
                                         backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.extract_high = backbone.layer4

        self.jaf_extractor_64 = PixelFeatureSampler()
        self.jaf_extractor_32 = PixelFeatureSampler()
        self.jaf_extractor_16 = PixelFeatureSampler()

        # regression-head
        self.regression_head = SimpleHead(self.channel)

    def forward(self, x):

        feat_64 = self.extract_mid(x)
        feat_32 = self.layer2(feat_64)
        feat_16 = self.layer3(feat_32)
        feat_high = self.extract_high(feat_16)

        coord_xyz, coord_uvd, pose_axis_angle, shape, mano_params = self.regression_head(feat_high)
        jaf_64 = self.jaf_extractor_64(coord_uvd[:, 778:, :2].detach(), feat_64).permute(0, 2, 1)
        jaf_32 = self.jaf_extractor_32(coord_uvd[:, 778:, :2].detach(), feat_32).permute(0, 2, 1)
        jaf_16 = self.jaf_extractor_16(coord_uvd[:, 778:, :2].detach(), feat_16).permute(0, 2, 1)
        jaf = torch.cat((jaf_64, jaf_32, jaf_16), dim=2)
        return coord_xyz, coord_uvd, pose_axis_angle, shape, mano_params, feat_high, jaf


@MODEL_REGISTRY.register()
class MultiViewNet(BaseModel):
    def __init__(self):
        super(MultiViewNet, self).__init__()
        self.sve = SingleViewNet()
        self.cvi = CrossViewInteraction(cfg.num_view, cfg.num_joint, self.sve.channel, 32)
        # counter
        self.batch_counter = 0
        self.start_batch = 0#40000 # float('inf')#0#40000 #float('inf')
        if cfg.finetune:
            self.start_batch = 0
        # loss
        self.coord_loss = nn.L1Loss()
        self.mvc_loss = MultiViewConsistencyLoss()

    def forward(self, x, targets=None, view_num=None):
        # for saving outputs
        outs = defaultdict(list)

        input_tensor = self.process_single_view_input(x)
        coord_xyz, coord_uvd, pose, shape, mano_params, feat_high, jaf = self.sve(input_tensor)
        coord_xyz_mvf, coord_uvd_mvf, pose_mvf, shape_mvf, cam_mvf = self.cvi(
            feat_high, jaf, coord_uvd[:, 778:], mano_params, view_num)


        # multi-view output
        outs['coord_uvd_mvf'].append(coord_uvd_mvf)
        outs['coord_xyz_mvf'].append(coord_xyz_mvf)
        outs['pose_mvf'].append(pose_mvf)
        outs['shape_mvf'].append(shape_mvf)
        outs['cam_mvf'].append(cam_mvf)
        # single-view output
        outs['coord_xyz'].append(coord_xyz)
        outs['coord_uvd'].append(coord_uvd)
        outs['pose'].append(pose)
        outs['shape'].append(shape)
        outs['cam'].append(mano_params[:, 16*6+10:])

        if targets is not None:
            return self.compute_loss(outs, targets)
        else:
            return self.process_output(outs)

    def compute_loss(self, outputs, targets):
        loss = {}
        error = {}
        mesh_pose_uvd = targets['mesh_pose_uvd'].view(-1, 799, 3).to(cfg.device)
        mesh_pose_uvd_gt = targets['mesh_pose_uvd_gt'].to(cfg.device).view(-1, 799, 3)
        mesh_pose_xyz = targets['mesh_pose_xyz'].view(-1, 799, 3).to(cfg.device)
        confidence = targets['confidence'].to(cfg.device).view(-1, 21, 1)

        if cfg.use_2D_GT: 
            # use 2D ground-truth for training
            confidence = torch.ones_like(confidence).to(cfg.device)
            mesh_pose_uvd = mesh_pose_uvd_gt

        # get relative rotation
        view_num = cfg.num_view
        if cfg.use_GTR:
            R = targets['rotation'].reshape(-1, view_num, 3, 3)[:, :, :3, :3]
        else:
            R = None
        for i in range(cfg.num_stage):
            if self.batch_counter >= self.start_batch or not self.training:
                mvc_mvf_loss_list, avg_fuse = self.mvc_loss(outputs['coord_xyz_mvf'][i], view_num, R)
                loss[f'mvc_mvf_{i}'] = sum(mvc_mvf_loss_list) / len(mvc_mvf_loss_list)
                loss[f'coord_xyz_dist_{i}'] = self.coord_loss(outputs['coord_xyz'][i], avg_fuse.reshape(-1, 799, 3).detach())
            else:
                avg_fuse = outputs['coord_xyz_mvf'][i].clone()
                
            self.batch_counter += 1

            loss[f'coord_uv_{i}'] = self.coord_loss(outputs['coord_uvd'][i][:, 778:, :2] * confidence,
                                                    mesh_pose_uvd[:, 778:, :2] * confidence)
            # prior loss
            loss[f'prior_shape_{i}'] = self.coord_loss(outputs['shape'][i], torch.zeros_like(outputs['shape'][i]).to(cfg.device)) # * 100
            loss[f'prior_pose_{i}'] = self.coord_loss(outputs['pose'][i][:, 3:], torch.zeros_like(outputs['pose'][i][:, 3:]).to(cfg.device)) * 0.01

            # # mvf
            loss[f'coord_uv_mvf_{i}'] = self.coord_loss(outputs['coord_uvd_mvf'][i][:, 778:, :2] * confidence, mesh_pose_uvd[:, 778:, :2] * confidence)
            loss[f'prior_pose_mvf_{i}'] = self.coord_loss(outputs['pose_mvf'][i][:, 3:], torch.zeros_like(outputs['pose'][i][:, 3:]).to(cfg.device)) * 0.01

            # get error
            # scale normalization
            center_gt = mesh_pose_xyz[:, 778:][:, 9:10]
            center_pred = outputs['coord_xyz'][i][:, 778:][:, 9:10]
            scale_pred = torch.norm(outputs['coord_xyz'][i][:, 778+9] - outputs['coord_xyz'][i][:, 778+0], p=2, dim=-1)
            scale_gt = torch.norm(mesh_pose_xyz[:, 778+9] - mesh_pose_xyz[:, 778+0], p=2, dim=-1)
            center_pred_mv = outputs['coord_xyz_mvf'][i][:, 778:][:, 9:10]
            scale_pred_mv = torch.norm(outputs['coord_xyz_mvf'][i][:, 778+9] - outputs['coord_xyz_mvf'][i][:, 778+0], p=2, dim=-1)

            avg_fuse = avg_fuse.reshape(-1, 799, 3) 
            center_pred_mv_avg = avg_fuse[:, 778:][:, 9:10]
            scale_pred_mv_avg = torch.norm(avg_fuse[:, 778+9] - avg_fuse[:, 778+0], p=2, dim=-1)
            avg_fuse = (avg_fuse - center_pred_mv_avg) * scale_gt.unsqueeze(1).unsqueeze(1) / scale_pred_mv_avg.unsqueeze(1).unsqueeze(1)

            outputs['coord_xyz_mvf'][i] = (outputs['coord_xyz_mvf'][i] - center_pred_mv) * scale_gt.unsqueeze(1).unsqueeze(1) / scale_pred_mv.unsqueeze(1).unsqueeze(1)
            outputs['coord_xyz'][i] = (outputs['coord_xyz'][i] - center_pred) * scale_gt.unsqueeze(1).unsqueeze(1) / scale_pred.unsqueeze(1).unsqueeze(1)
            mesh_pose_xyz = mesh_pose_xyz - center_gt

            # get training error
            # 2d
            error[f'OP_and_GT_{i}'] = (torch.norm(
                (mesh_pose_uvd[:, 778:, :2] + 1) * cfg.input_img_shape[0] / 2 - 
                (mesh_pose_uvd_gt[:, 778:, :2] + 1) * cfg.input_img_shape[0] / 2, p=2, dim=2)).detach().mean()

            # 2d
            error[f'pose_uv_mvf_{i}'] = (torch.norm(
                (outputs['coord_uvd_mvf'][i][:, 778:, :2] + 1) * cfg.input_img_shape[0] / 2 - 
                (mesh_pose_uvd_gt[:, 778:, :2] + 1) * cfg.input_img_shape[0] / 2, p=2, dim=2)).detach().mean()
            error[f'pose_uv_{i}'] = (torch.norm(
                (outputs['coord_uvd'][i][:, 778:, :2] + 1) * cfg.input_img_shape[0] / 2 - 
                (mesh_pose_uvd_gt[:, 778:, :2] + 1) * cfg.input_img_shape[0] / 2, p=2, dim=2)).detach().mean()

            # 3d
            error[f'pose_xyz_mvf_avg_{i}'] = (torch.norm(
                avg_fuse.reshape(-1, 799, 3)[:, 778:] * cfg.bbox_3d_size / 2 - 
                mesh_pose_xyz[:, 778:] * cfg.bbox_3d_size / 2, p=2, dim=2) * 1000).detach().mean()
            error[f'pose_xyz_mvf_{i}'] = (torch.norm(
                outputs['coord_xyz_mvf'][i][:, 778:] * cfg.bbox_3d_size / 2 - 
                mesh_pose_xyz[:, 778:] * cfg.bbox_3d_size / 2, p=2, dim=2) * 1000).detach().mean()
            error[f'pose_xyz_{i}'] = (torch.norm(
                outputs['coord_xyz'][i][:, 778:] * cfg.bbox_3d_size / 2 - 
                mesh_pose_xyz[:, 778:] * cfg.bbox_3d_size / 2, p=2, dim=2) * 1000).detach().mean()
            
            for view_idx in range(cfg.num_view):
                pred = avg_fuse.reshape(-1, 799, 3)[:, 778:].reshape(-1, view_num, 21, 3)
                gt = mesh_pose_xyz[:, 778:].reshape(-1, view_num, 21, 3)
                error[f'pose_xyz_mvf_{i}_view_{view_idx}'] = (torch.norm(
                    pred[:, view_idx] * cfg.bbox_3d_size / 2 - 
                    gt[:, view_idx] * cfg.bbox_3d_size / 2, p=2, dim=2) * 1000).detach().mean()

            aligned_xyz, (_, pred_R, _) = batch_compute_similarity_transform_torch(outputs['coord_xyz'][i][:, 778:].permute(0, 2, 1), mesh_pose_xyz[:, 778:].permute(0, 2, 1))
            aligned_xyz = aligned_xyz.permute(0, 2, 1)
            error[f'pose_alinged_xyz'] = (torch.norm(
                aligned_xyz * 0.4 / 2 - 
                mesh_pose_xyz[:, 778:] * 0.4 / 2, p=2, dim=2) * 1000).detach().mean()
            
            aligned_xyz_mvf, (_, pred_R, _) = batch_compute_similarity_transform_torch(outputs['coord_xyz_mvf'][i][:, 778:].permute(0, 2, 1), mesh_pose_xyz[:, 778:].permute(0, 2, 1))
            aligned_xyz_mvf = aligned_xyz_mvf.permute(0, 2, 1)
            error[f'pose_alinged_mvf_xyz'] = (torch.norm(
                aligned_xyz_mvf * 0.4 / 2 - 
                mesh_pose_xyz[:, 778:] * 0.4 / 2, p=2, dim=2) * 1000).detach().mean()
            aligned_xyz_fuse, (_, pred_R, _) = batch_compute_similarity_transform_torch(avg_fuse.reshape(-1, 799, 3)[:, 778:].permute(0, 2, 1), mesh_pose_xyz[:, 778:].permute(0, 2, 1))
            aligned_xyz_fuse = aligned_xyz_fuse.permute(0, 2, 1)
            error[f'pose_alinged_fuse_xyz'] = (torch.norm(
                aligned_xyz_fuse * 0.4 / 2 - 
                mesh_pose_xyz[:, 778:] * 0.4 / 2, p=2, dim=2) * 1000).detach().mean()

        return loss, error

    def process_output(self, outputs):
        outs = {
            'coord_xyz': sum(outputs['coord_xyz']) / len(outputs['coord_xyz']),
            'coord_uvd': sum(outputs['coord_uvd']) / len(outputs['coord_uvd']),
            'coord_xyz_mv': sum(outputs['coord_xyz_mvf']) / len(outputs['coord_xyz_mvf']),
            'coord_uvd_mv': sum(outputs['coord_uvd_mvf']) / len(outputs['coord_uvd_mvf']),
            'camera': sum(outputs['cam']) / len(outputs['cam']),
            'camera_mv': sum(outputs['cam_mvf']) / len(outputs['cam_mvf']),
        }


        outs['coord_uvd'][:, :, :2] = (outs['coord_uvd'][:, :, :2] + 1) * (cfg.input_img_shape[0] / 2)
        outs['coord_uvd'][:, :, 2] = outs['coord_uvd'][:, :, 2] * (cfg.bbox_3d_size / 2)
        outs['coord_xyz'] = outs['coord_xyz'] * (cfg.bbox_3d_size / 2)
        
        outs['coord_uvd_mv'][:, :, :2] = (outs['coord_uvd_mv'][:, :, :2] + 1) * (cfg.input_img_shape[0] / 2)
        outs['coord_uvd_mv'][:, :, 2] = outs['coord_uvd_mv'][:, :, 2] * (cfg.bbox_3d_size / 2)
        outs['coord_xyz_mv'] = outs['coord_xyz_mv'] * (cfg.bbox_3d_size / 2)
        return outs

