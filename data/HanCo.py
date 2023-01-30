import sys
from numpy.random.mtrand import f
from torch.utils.data import Dataset
from data.processing import load_img, augmentation, xyz2uvd, uvd2xyz
import os
import torch
import glob
import json
import typing
import numpy as np
import collections
import torchvision.transforms as standard
from config import cfg
import random
from data.augmentation import *
from data.dataset import DATASET_REGISTRY


def parse_path(file_path):
    # return sequence name and frame name
    parts = file_path.split('/')
    return parts[-2], parts[-1][:-5]

def get_seq_cam_frame(file_path):
    # return sequence, camera, and frame.
    parts = file_path.split('/')
    return parts[-3], parts[-2], parts[-1][:-5]

@DATASET_REGISTRY.register()
class HanCo(Dataset):
    def __init__(self, root, split):
        super(HanCo, self).__init__()
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = standard.Compose([standard.ToTensor(), standard.Normalize(*mean_std)])
        self.select_view_idx = cfg.view_idx
        self.cam_num = 8
        self.data_split = split

        if self.data_split == 'train':
            self.image_type = 'rgb'
        else:
            self.image_type = 'rgb_merged'
            # self.image_type = 'rgb'

        # home rgb set.
        self.homo_rgb_image_filenames = set(glob.glob(os.path.join(root,self.image_type,'**','*.jpg'), recursive=True))
        # get keypoints list.
        keypoints_filenames_all_cams = glob.glob(os.path.join(root,'xyz','**','*.json'), recursive=True)
        keypoints_filenames_all_cams.sort()
        self.keypoints_filenames = [v for v in keypoints_filenames_all_cams for _ in range(self.cam_num)]
        # get clean rbg images list and camera parameters list.
        self.rgb_image_filenames = []
        self.camera_filenames = {}
        self.select_keypoints_filenames = []
        self.all_image_type = []
        for kps_idx, file_path in enumerate(self.keypoints_filenames):
            seq, frame = parse_path(file_path)
            train_data_ratio = 1
            # train_data_ratio = 0.5
            # 100%
            if self.data_split == 'train' and int(seq) >= 1200 * train_data_ratio:
                continue

            if self.data_split != 'train' and int(seq) < 1200:
                continue

            cam_idx = kps_idx % self.cam_num

            if os.path.join(root, self.image_type, seq, 'cam' + str(cam_idx), frame + '.jpg') not in self.homo_rgb_image_filenames:
                continue

            # add gt xyz
            self.select_keypoints_filenames.append(file_path)

            self.rgb_image_filenames.append(os.path.join(root, self.image_type, seq, 'cam' + str(cam_idx), frame + '.jpg'))

            frame_idx = len(self.rgb_image_filenames) - 1
            self.camera_filenames[frame_idx] = {}
            self.camera_filenames[frame_idx]['K'] = os.path.join(root, 'calib', seq, frame + '.json')
            self.camera_filenames[frame_idx]['cam_idx'] = cam_idx
        self.keypoints_filenames = self.select_keypoints_filenames
        print('Total {} image number is {}.'.format(self.data_split, len(self.rgb_image_filenames)))

    def __len__(self) -> int:
        return len(self.rgb_image_filenames) // self.cam_num

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:

        inputs = collections.defaultdict(list)
        targets = collections.defaultdict(list)
        meta_info = collections.defaultdict(list)

        start_idx = index * self.cam_num
        for offset in range(self.cam_num):
            index = start_idx + offset
            # get image
            rgb_img_filename = self.rgb_image_filenames[index]

            # load openPose
            result_file = rgb_img_filename.replace(self.image_type, 'rgb_2D_keypoints').replace('jpg', 'json')

            with open(result_file, 'r') as fi:
                kps, cfd = json.load(fi)
            kps = np.array(kps, dtype=np.float32).reshape(21, 2)
            cfd = np.array(cfd, dtype=np.float32).reshape(21, 1)
            keypoints_uv = kps

            img = load_img(rgb_img_filename)
            ori_img = img.copy()

            if cfg.aug and self.data_split == 'train':
                img = blurAugmentation(img, 0.5)
                img = occlusionAugmentation(img, 0.5)

            # get keypoints
            keypoints_xyz = self._json_load(self.keypoints_filenames[index])
            # get camera parameters.
            cam_idx = self.camera_filenames[index]['cam_idx']
            K_list = self._json_load(self.camera_filenames[index]['K'])
            extrinsic = np.array(K_list['M'][cam_idx])
            intrinsic = np.array(K_list['K'][cam_idx])
            K, w2cam, keypoints_xyz = [np.array(x) for x in [K_list['K'][cam_idx], K_list['M'][cam_idx], keypoints_xyz]]
            
            # world2camera
            keypoints_xyz = keypoints_xyz.reshape(21, 3)
            world_coord = keypoints_xyz.copy()
            keypoints_xyz = np.matmul(keypoints_xyz, w2cam[:3, :3].T) + w2cam[:3, 3][None]

            keypoints_uv_gt = xyz2uvd(keypoints_xyz, K)
            keypoints_uv = np.concatenate((keypoints_uv, keypoints_uv_gt[:, 2:3].copy()), axis=1)

            # augmentation
            center_x = 112
            center_y = 112

            # cropping augmentation
            if self.data_split == 'train':
                size = np.random.randint(120, 160)
                random_noise_x = np.random.randint(-20, 20)
                random_noise_y = np.random.randint(-20, 20)
                center_x += random_noise_x
                center_y += random_noise_y
            else:
                size = 130
            bbox = [center_x-size//2, center_y-size//2, size, size]
            img, img2bb_trans, bb2img_trans, rot, _ = augmentation(img, bbox, self.data_split,
                                                        exclude_flip=True, out_img_shape=cfg.input_img_shape)  # FreiHAND dataset only contains right hands. do not perform flip aug.
            crop_img = img.copy()

            # affine transform x,y coordinates. root-relative depth
            keypoints_xy1 = np.concatenate((keypoints_uv[:, :2], np.ones_like(keypoints_uv[:, :1])), 1)
            keypoints_uv[:, :2] = np.dot(img2bb_trans, keypoints_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]

            keypoints_xy1_gt = np.concatenate((keypoints_uv_gt[:, :2], np.ones_like(keypoints_uv_gt[:, :1])), 1)
            keypoints_uv_gt[:, :2] = np.dot(img2bb_trans, keypoints_xy1_gt.transpose(1, 0)).transpose(1, 0)[:, :2]

            # 3D augmentation
            # 3D data rotation augmentation
            rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                                    [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                                    [0, 0, 1]], dtype=np.float32)

            # modify intrinsic 
            intrinsic = np.dot(np.concatenate((img2bb_trans, np.array([[0, 0, 1]], dtype=np.float32)), axis=0), intrinsic.copy())
            
            # modify extrinsic rotation
            rotation = np.dot(rot_aug_mat, extrinsic[:3, :3])

            # coordinates normalize
            mano_coord_img = np.zeros(shape=(799, 3))
            mano_coord_img[778:, :3] = keypoints_uv
            root_joint_depth = keypoints_uv[cfg.root_idx, 2:3]
            mano_coord_img[:, 2] = mano_coord_img[:, 2] - root_joint_depth
            mano_coord_img[:, 2] = mano_coord_img[:, 2] / (cfg.bbox_3d_size / 2)
            mano_coord_img[:, 0] = mano_coord_img[:, 0] / (cfg.input_img_shape[0] / 2) - 1
            mano_coord_img[:, 1] = mano_coord_img[:, 1] / (cfg.input_img_shape[1] / 2) - 1

            # uvd2xyz
            keypoints_xyz = keypoints_xyz - keypoints_xyz[cfg.root_idx]
            keypoints_xyz = np.dot(rot_aug_mat, keypoints_xyz.transpose(1, 0)).transpose(1, 0)
            mano_coord_cam = np.zeros(shape=(799, 3))
            mano_coord_cam[778:, :3] = keypoints_xyz
            mano_coord_cam = mano_coord_cam  / (cfg.bbox_3d_size / 2)


            mano_coord_img_gt = np.zeros(shape=(799, 3))
            mano_coord_img_gt[778:, :3] = keypoints_uv_gt
            root_joint_depth_gt = keypoints_uv_gt[cfg.root_idx, 2:3]
            mano_coord_img_gt[:, 2] = mano_coord_img_gt[:, 2] - root_joint_depth_gt
            mano_coord_img_gt[:, 2] = mano_coord_img_gt[:, 2] / (cfg.bbox_3d_size / 2)
            mano_coord_img_gt[:, 0] = mano_coord_img_gt[:, 0] / (cfg.input_img_shape[0] / 2) - 1
            mano_coord_img_gt[:, 1] = mano_coord_img_gt[:, 1] / (cfg.input_img_shape[1] / 2) - 1

            # 
            keypoints_xyz = mano_coord_cam
            keypoints_uv = mano_coord_img 
            keypoints_uv_gt = mano_coord_img_gt

            img = self.transform(img.astype(np.uint8))

            inputs['img'].append(img)
            inputs['extrinsic'].append(np.float32(extrinsic))
            inputs['intrinsic'].append(np.float32(intrinsic))
            inputs['rotation'].append(np.float32(rotation))
            targets['mesh_pose_uvd'].append(keypoints_uv)
            targets['mesh_pose_xyz'].append(keypoints_xyz)
            targets['mesh_pose_uvd_gt'].append(keypoints_uv_gt)
            targets['intrinsic'].append(intrinsic)
            targets['rotation'].append(rotation)
            targets['extrinsic'].append(extrinsic)
            targets['img2bb_trans'].append(img2bb_trans)
            targets['bb2img_trans'].append(bb2img_trans)
            targets['world_coord'].append(world_coord)
            targets['confidence'].append(cfd)
            targets['proj_matrix'].append(np.matmul(intrinsic, extrinsic[:3]))
            targets['crop_img'].append(crop_img)
            targets['ori_img'].append(ori_img)

        inputs = {k: np.stack(v, axis=0) for k, v in inputs.items()}
        targets = {k: np.float32(np.stack(v, axis=0)) for k, v in targets.items() if k != 'result_file' and k != 'kps_file_name'}
        meta_info = {k: np.float32(np.stack(v, axis=0)) for k, v in meta_info.items()}


        # view select
        inputs = {k: v[self.select_view_idx] for k, v in inputs.items()}
        targets = {k: v[self.select_view_idx] for k, v in targets.items()}
        meta_info = {k: v[self.select_view_idx] for k, v in meta_info.items()}

        # # add random camera order to obtain camera input equivariance
        random_order = list(range(len(self.select_view_idx)))
        random.shuffle(random_order)
        inputs = {k: v[random_order] for k, v in inputs.items()}
        targets = {k: v[random_order] for k, v in targets.items()}
        meta_info = {k: v[random_order] for k, v in meta_info.items()}
        return inputs, targets, meta_info


    def _json_load(self, p):
        with open(p, 'r') as fi:
            d = json.load(fi)
        return d

