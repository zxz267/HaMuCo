import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
import math
import random
from utils.mano import MANO
import torchgeometry as tgm
from data.processing import orthgonalProj


class SoftHeatmap(nn.Module):
    def __init__(self, size, kp_num):
        super(SoftHeatmap, self).__init__()
        self.size = size
        self.beta = nn.Conv2d(kp_num, kp_num, 1, 1, 0, groups=kp_num, bias=False)
        self.wx = torch.arange(0.0, 1.0 * self.size, 1).view([1, self.size]).repeat([self.size, 1])
        self.wy = torch.arange(0.0, 1.0 * self.size, 1).view([self.size, 1]).repeat([1, self.size])
        self.wx = nn.Parameter(self.wx, requires_grad=False)
        self.wy = nn.Parameter(self.wy, requires_grad=False)

    def forward(self, x):
        s = list(x.size())
        scoremap = self.beta(x)
        scoremap = scoremap.view([s[0], s[1], s[2] * s[3]])
        scoremap = F.softmax(scoremap, dim=2)
        scoremap = scoremap.view([s[0], s[1], s[2], s[3]])
        scoremap_x = scoremap.mul(self.wx)
        scoremap_x = scoremap_x.view([s[0], s[1], s[2] * s[3]])
        soft_argmax_x = torch.sum(scoremap_x, dim=2)
        scoremap_y = scoremap.mul(self.wy)
        scoremap_y = scoremap_y.view([s[0], s[1], s[2] * s[3]])
        soft_argmax_y = torch.sum(scoremap_y, dim=2)
        keypoint_uv = torch.stack([soft_argmax_x, soft_argmax_y], dim=2)
        return keypoint_uv, scoremap

class GraphConv(nn.Module):
    def __init__(self, num_joint, in_features, out_features):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.adj = nn.Parameter(torch.eye(num_joint).float().to(cfg.device), requires_grad=True)

    def laplacian(self, A_hat):
        D_hat = torch.sum(A_hat, 1, keepdim=True) + 1e-5
        L = 1 / D_hat * A_hat
        return L

    def forward(self, x):
        batch = x.size(0)
        A_hat = self.laplacian(self.adj)
        A_hat = A_hat.unsqueeze(0).repeat(batch, 1, 1)
        out = self.fc(torch.matmul(A_hat, x))
        return out

class GraphRegression(nn.Module):
    def __init__(self, node_num, in_dim, out_dim, layer_num=2, last=True):
        super(GraphRegression, self).__init__()
        self.num_node = node_num
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = nn.LeakyReLU(0.1)
        self.reg = nn.Sequential()
        self.reg.add_module('ln', nn.LayerNorm(self.in_dim))
        for i in range(layer_num-1):
            self.reg.add_module('gcn_i', GraphConv(node_num, self.in_dim, self.in_dim))
            self.reg.add_module('activate_i', self.activation)
        self.reg.add_module(f'dp', nn.Dropout(0.1))
        self.reg.add_module(f'gcn_{layer_num-1}', GraphConv(node_num, self.in_dim, self.out_dim))
        if not last:
            self.reg.add_module(f'activate_{layer_num-1}', self.activation)

    def forward(self, graph, shortcut=False):
        in_graph = graph
        out_graph = self.reg(graph)
        if shortcut:
            assert in_graph.shape[2] == out_graph.shape[2]
            return out_graph + in_graph
        else:
            return out_graph

class MLP_res_block(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, in_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _ff_block(self, x):
        x = self.fc2(self.dropout1(F.relu(self.fc1(x))))
        return self.dropout2(x)

    def forward(self, x):
        x = x + self._ff_block(self.layer_norm(x))
        return x

class SelfAttn(nn.Module):
    def __init__(self, f_dim, hid_dim=None, n_heads=4, d_q=None, d_v=None, dropout=0.1):
        super().__init__()
        if d_q is None:
            d_q = f_dim // n_heads
        if d_v is None:
            d_v = f_dim // n_heads
        if hid_dim is None:
            hid_dim = f_dim

        self.n_heads = n_heads
        self.d_q = d_q
        self.d_v = d_v
        self.norm = d_q ** 0.5
        self.f_dim = f_dim

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.w_qs = nn.Linear(f_dim, n_heads * d_q)
        self.w_ks = nn.Linear(f_dim, n_heads * d_q)
        self.w_vs = nn.Linear(f_dim, n_heads * d_v)

        self.layer_norm = nn.LayerNorm(f_dim, eps=1e-6)
        self.fc = nn.Linear(n_heads * d_v, f_dim)

        self.ff = MLP_res_block(f_dim, hid_dim, dropout)

    def self_attn(self, x, valid=None, mask=False):
        BS, V, f = x.shape

        q = self.w_qs(x).view(BS, -1, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        k = self.w_ks(x).view(BS, -1, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        v = self.w_vs(x).view(BS, -1, self.n_heads, self.d_v).transpose(1, 2)  # BS x h x V x v

        attn = torch.matmul(q, k.transpose(-1, -2)) / self.norm  # bs, h, V, V

        if mask and self.training:
            # mask = torch.rand(168, 168)
            # mask = torch.where(mask >= 0.5, float(0), float('-inf'))
            use_view_num = random.randint(1, cfg.num_view-1)
            joint_num = 21
            mask = torch.zeros(joint_num * cfg.num_view, joint_num * cfg.num_view)
            for view_idx in range(cfg.num_view):
                use_view_set = random.sample([i for i in range(cfg.num_view) if i != view_idx], cfg.num_view - use_view_num)
                use_view_set.sort()
                for use_idx in use_view_set:
                    mask[view_idx*joint_num:(view_idx+1)*joint_num, use_idx*joint_num:(use_idx+1)*joint_num] = torch.zeros_like(mask[view_idx*joint_num:(view_idx+1)*joint_num, use_idx*joint_num:(use_idx+1)*joint_num]) + float('-inf')
            # mask_joint = torch.rand(joint_num * cfg.num_view, joint_num * cfg.num_view)
            # mask_joint = torch.where(mask_joint > 0.5, float(0), float('-inf'))
            # mask = mask_joint + mask
            mask = mask.unsqueeze(0).unsqueeze(1).repeat(attn.shape[0], attn.shape[1], 1, 1)
            attn = attn + mask.to(cfg.device)
        
        if valid is not None:
            joint_num = 21
            valid = valid.view(-1, cfg.num_view)
            batch = valid.shape[0]
            valid = valid.view(batch, 1, cfg.num_view).repeat(1, joint_num, 1).permute(0, 2, 1).reshape(batch, 1, -1)
            valid = torch.where(valid == 0, float(-2**32+1), float(0)).repeat(1, cfg.num_view * joint_num, 1)
            valid = valid.permute(0, 2, 1) + valid
            valid = valid.unsqueeze(1)
            attn = attn + valid
            

        attn = F.softmax(attn, dim=-1)  # bs, h, V, V
        attn = self.dropout1(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(BS, V, -1)
        out = self.dropout2(self.fc(out))
        return out

    def forward(self, x, valid=None, mask=False):
        BS, V, f = x.shape
        assert f == self.f_dim

        x = x + self.self_attn(self.layer_norm(x), valid, mask)
        x = self.ff(x)

        return x


class PixelFeatureSampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, point_uv, s_feat):
        return torch.nn.functional.grid_sample(s_feat, point_uv.unsqueeze(2), align_corners=True)[..., 0]



class ManoDecoder(nn.Module):
    def __init__(self):
        super(ManoDecoder, self).__init__()
        self.mano_layer = MANO('right').layer.to(cfg.device)
        self.joint_regressor = MANO().joint_regressor
        self.root_joint_idx = cfg.root_idx

    def rot6d_to_rotmat(self, x):
        x = x.reshape(-1, 3, 2)
        a1 = x[:, :, 0]
        a2 = x[:, :, 1]
        b1 = F.normalize(a1)
        b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
        b3 = torch.cross(b1, b2)
        return torch.stack((b1, b2, b3), dim=-1)

    def forward(self, pose, shape, cam):
        batch = pose.shape[0]
        # transform rot-6d to angle-axis
        pose = self.rot6d_to_rotmat(pose)
        # pose = kornia.geometry.conversions.rotation_matrix_to_angle_axis(pose).reshape(batch, -1)
        pose = torch.cat([pose, torch.zeros((pose.shape[0], 3, 1)).to(cfg.device).float()], 2)
        pose = tgm.rotation_matrix_to_angle_axis(pose).reshape(batch, -1)
        # get coordinates from MANO layer
        mano_mesh_cam, _ = self.mano_layer(pose, shape)
        # mm to m
        mano_mesh_cam = mano_mesh_cam / 1000
        # get pose joints
        mano_joint_cam = torch.bmm(
            torch.from_numpy(self.joint_regressor).to(cfg.device)[None, :, :].repeat(batch, 1, 1), mano_mesh_cam)
        coord_xyz = torch.cat((mano_mesh_cam, mano_joint_cam), dim=1)
        # root-relative
        coord_xyz = coord_xyz - mano_joint_cam[:, self.root_joint_idx, None]
        # project xy to uv
        coord_uv = orthgonalProj(coord_xyz[:, :, :2].clone(), cam[:, 0:1].unsqueeze(1), cam[:, 1:].unsqueeze(1))
        # normalization
        coord_xyz = coord_xyz / (cfg.bbox_3d_size / 2)
        coord_uv = coord_uv / (cfg.input_img_shape[0] // 2) - 1
        coord_uvd = torch.cat((coord_uv, coord_xyz[:, :, 2:3]), dim=2)
        return coord_xyz, coord_uvd, pose, shape, cam

class ManoRegressor(nn.Module):
    def __init__(self, in_dim):
        super(ManoRegressor, self).__init__()
        self.in_dim = in_dim
        self.mano_decoder = ManoDecoder()
        self.regressor = nn.Sequential(
            nn.Linear(in_dim + 6*16 + 10 + 3, 512),
            nn.GroupNorm(8, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, cfg.num_params)
        )

    def forward(self, feat, prev_mano_params):
        batch, feat_dim = feat.shape
        prev_mano_params = prev_mano_params.detach()
        updated_param = self.regressor(torch.cat((feat, prev_mano_params), axis=1)) + prev_mano_params
        cam = updated_param[:, 16*6+10:]
        cam = torch.cat((F.relu(cam[:, 0:1]), cam[:, 1:]), dim=1).view(batch, 3)
        coord_xyz, coord_uvd, pose, shape, cam = self.mano_decoder(
            updated_param[:, :16*6], 
            prev_mano_params[:, 16*6:16*6+10], 
            cam)
        return coord_xyz, coord_uvd, pose, shape, cam

        

def make_linear_layers(feat_dims, relu_final=True, use_bn=False):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_conv1d_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv1d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_deconv_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.ConvTranspose2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)



class SAIGB(nn.Module):
    def __init__(self, backbone_channels, num_FMs, feature_size, num_kps, position_embedding=False):
        super(SAIGB, self).__init__()
        self.backbone_channels = backbone_channels
        self.feature_size = feature_size
        self.num_kps = num_kps
        self.num_FMs = num_FMs
        self.group = nn.Sequential(
            nn.Conv2d(self.backbone_channels, self.num_FMs * self.num_kps, 1),
            nn.LeakyReLU(0.1)
        )
        if position_embedding:
            self.position_embeddings = nn.Embedding(self.num_kps, self.feature_size * self.num_FMs)
        self.use_position_embedding = position_embedding

    def forward(self, x):
        init_graph = self.group(x).reshape(-1, self.num_kps, self.feature_size * self.num_FMs)
        if self.use_position_embedding:
            position_ids = torch.arange(self.num_kps, dtype=torch.long, device=cfg.device)
            position_ids = position_ids.unsqueeze(0).repeat(x.shape[0], 1)
            position_embeddings = self.position_embeddings(position_ids)
            init_graph += position_embeddings
        return init_graph