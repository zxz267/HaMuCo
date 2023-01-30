import numpy as np
import cv2
import random
from config import cfg
import json
import os
from PIL import Image
import time
import torch
from zipfile import ZipFile
from io import BytesIO

def orthgonalProj(xy, scale, transl, img_size=256):
    scale = scale * img_size
    transl = transl * img_size / 2 + img_size / 2
    return xy * scale + transl

def perspectiveProj(xyz, K):
    uv = torch.matmul(K, xyz.permute(0, 2, 1)).permute(0, 2, 1)
    return uv[:, :, :2] / uv[:, :, -1:]

def perspectiveProjNumpy(xyz, K):
    uv = np.matmul(K, xyz.transpose(0, 2, 1)).transpose(0, 2, 1)
    return uv[:, :, :2] / uv[:, :, -1:]
    
def batch_torch_uvd2xyz(uvd, camera):
    # camera: [batch, 3, 3]
    f = torch.cat((camera[:, 0, 0:1], camera[:, 1, 1:2]), dim=1)
    c = camera[:, 0:2, 2]
    x = (uvd[:, :, 0:1] - c[:, None, 0:1]) / f[:, None, 0:1] * uvd[:, :, 2:3]
    y = (uvd[:, :, 1:2] - c[:, None, 1:2]) / f[:, None, 1:2] * uvd[:, :, 2:3]
    z = uvd[:, :, 2:3]
    return torch.cat((x, y, z), dim=2)

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
    y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
    z = cam_coord[:,2]
    return np.stack((x,y,z),1)

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:,0] - c[0]) / f[0] * pixel_coord[:,2]
    y = (pixel_coord[:,1] - c[1]) / f[1] * pixel_coord[:,2]
    z = pixel_coord[:,2]
    return np.stack((x,y,z),1)

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def cam2world(cam_coord, R, t):
    world_coord = np.dot(np.linalg.inv(R), (cam_coord - t.reshape(1,3)).transpose(1,0)).transpose(1,0)
    return world_coord

def cv2pil(cv_img):
    return Image.fromarray(cv2.cvtColor(np.uint8(cv_img), cv2.COLOR_BGR2RGB))

def uvd2xyz(uvd, K):
    fx, fy, fu, fv = K[0, 0], K[0, 0], K[0, 2], K[1, 2]
    xyz = np.zeros_like(uvd, np.float32)
    xyz[:, 0] = (uvd[:, 0] - fu) * uvd[:, 2] / fx
    xyz[:, 1] = (uvd[:, 1] - fv) * uvd[:, 2] / fy
    xyz[:, 2] = uvd[:, 2]
    return xyz

def xyz2uvd(xyz, K):
    fx, fy, fu, fv = K[0, 0], K[0, 0], K[0, 2], K[1, 2]
    uvd = np.zeros_like(xyz, np.float32)
    uvd[:, 0] = (xyz[:, 0] * fx / xyz[:, 2] + fu)
    uvd[:, 1] = (xyz[:, 1] * fy / xyz[:, 2] + fv)
    uvd[:, 2] = xyz[:, 2]
    return uvd

def get_focal_pp(K):
    """ Extract the camera parameters that are relevant for an orthographic assumption. """
    focal = [K[0, 0], K[1, 1]]
    pp = K[:2, 2]
    return focal, pp

""" General util functions. """
def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg


def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]

""" Dataset related functions. """
def db_size(set_name):
    """ Hardcoded size of the datasets. """
    if set_name == 'training':
        return 32560  # number of unique samples (they exists in multiple 'versions')
    elif set_name == 'evaluation':
        return 3960
    else:
        assert 0, 'Invalid choice.'

def load_db_annotation(base_path, set_name):
    assert set_name in ['training', 'evaluation'], 'mode error'

    print('Loading FreiHAND dataset index ...')
    t = time.time()
    if set_name == 'training':
        # assumed paths to data containers
        k_path = os.path.join(base_path, '%s_K.json' % set_name)
        xyz_path = os.path.join(base_path, '%s_xyz.json' % set_name)
        scale_path = os.path.join(base_path, '%s_scale.json' % set_name)
        # vert_path = os.path.join(base_path, '%s_verts.json' % set_name)
        mano_path = os.path.join(base_path, '%s_mano.json' % set_name)

        # load if exist
        K_list = json_load(k_path)
        # vert_list = json_load(vert_path)
        mano_list = json_load(mano_path)
        xyz_list = json_load(xyz_path)
        scale_list = json_load(scale_path)

        # should have all the same length
        # assert len(K_list) == len(vert_list), 'Size mismatch.'
        assert len(K_list) == len(xyz_list), 'Size mismatch.'
        assert len(K_list) == len(scale_list), 'Size mismatch.'

        print('Loading of %d samples done in %.2f seconds' % (len(K_list), time.time()-t))
        # return list(zip(K_list, vert_list, xyz_list, scale_list))
        return list(zip(K_list, mano_list, xyz_list, scale_list))
    else:
        # assumed paths to data containers
        k_path = os.path.join(base_path, '%s_K.json' % set_name)
        scale_path = os.path.join(base_path, '%s_scale.json' % set_name)

        # load if exist
        K_list = json_load(k_path)
        scale_list = json_load(scale_path)

        # should have all the same length
        assert len(K_list) == len(scale_list), 'Size mismatch.'

        print('Loading of %d samples done in %.2f seconds' % (len(K_list), time.time() - t))
        return list(zip(K_list, scale_list))

def pad2squre(img, color=None):
    if img.shape[0] > img.shape[1]:
        W = img.shape[0] - img.shape[1]
    else:
        W = img.shape[1] - img.shape[0]
    W1 = int(W / 2)
    W2 = W - W1
    if color is None:
        if img.shape[2] == 3:
            color = (0, 0, 0)
        else:
            color = 0
    if img.shape[0] > img.shape[1]:
        return cv2.copyMakeBorder(img, 0, 0, W1, W2, cv2.BORDER_CONSTANT, value=color)
    else:
        return cv2.copyMakeBorder(img, W1, W2, 0, 0, cv2.BORDER_CONSTANT, value=color)

def getSquare(img):
    if img.shape[0] != img.shape[1]:
        length = min(img.shape[0], img.shape[1])
        center = (img.shape[0] // 2, img.shape[1] // 2)
        x = center[0] - length // 2
        y = center[1] - length // 2
        return img[x:x+length, y:y+length]
    else:
        return img


def imcrop(img, center, crop_size):
    x1 = int(np.round(center[0]-crop_size))
    y1 = int(np.round(center[1]-crop_size))
    x2 = int(np.round(center[0]+crop_size))
    y2 = int(np.round(center[1]+crop_size))

    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
         img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)

    if img.ndim < 3: # for depth
        img_crop = img[y1:y2, x1:x2]
    else: # for rgb
        img_crop = img[y1:y2, x1:x2, :]

    trans = np.eye(3)
    trans[0, 2] = -x1
    trans[1, 2] = -y1

    return img_crop, trans

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    borderValue = [127, 127, 127]

    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                                 -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_CONSTANT, value=borderValue)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2

def convert_kp(keypoints):
    kp_dict = {0: 0, 1: 20, 2: 19, 3: 18, 4: 17, 5: 16, 6: 15, 7: 14, 8: 13, 9: 12, 10: 11, 11: 10,
               12: 9, 13: 8, 14: 7, 15: 6, 16: 5, 17: 4, 18: 3, 19: 2, 20: 1}

    keypoints_new = list()
    for i in range(21):
        if i in kp_dict.keys():
            pos = kp_dict[i]
            keypoints_new.append(keypoints[pos, :])

    return np.stack(keypoints_new, 0)

# def load_img(path, order='RGB', rgb_image=True):
#     if rgb_image:
#         img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
#         if not isinstance(img, np.ndarray):
#             raise IOError("Fail to read %s" % path)

#         if order == 'RGB':
#             img = img[:, :, ::-1].copy()
#     else:
#         img = cv2.imread(path, 0)
#     return img


def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order == 'RGB':
        img = img[:, :, ::-1].copy()
    return img

def load_img_from_zip(zip_file, name):
    # rgb
    # img = Image.open(BytesIO(zip_file.read(name)))
    # return np.asarray(img)
    return cv2.imdecode(np.frombuffer(zip_file.read(name), np.uint8), 1)[:, :, ::-1]

def get_bbox(joint_img, joint_valid):
    x_img, y_img = joint_img[:, 0], joint_img[:, 1]
    x_img = x_img[joint_valid == 1]
    y_img = y_img[joint_valid == 1]
    xmin = min(x_img)
    ymin = min(y_img)
    xmax = max(x_img)
    ymax = max(y_img)

    x_center = (xmin + xmax) / 2.
    width = xmax - xmin
    xmin = x_center - 0.5 * width * 1.2
    xmax = x_center + 0.5 * width * 1.2

    y_center = (ymin + ymax) / 2.
    height = ymax - ymin
    ymin = y_center - 0.5 * height * 1.2
    ymax = y_center + 0.5 * height * 1.2

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox

def process_bbox(bbox, img_width, img_height):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w * h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])
    else:
        return None

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    aspect_ratio = cfg.input_img_shape[1] / cfg.input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * 1.25
    bbox[3] = h * 1.25
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.

    return bbox

def get_aug_config(exclude_flip):
    scale_factor = (0.9, 1.1)
    rot_factor = 180
    color_factor = 0.2
    transl_factor = 10
    scale = np.random.rand() * (scale_factor[1] - scale_factor[0]) + scale_factor[0]
    rot = (np.random.rand() * 2 - 1) * rot_factor
    transl_x = (np.random.rand() * 2 - 1) * transl_factor
    transl_y = (np.random.rand() * 2 - 1) * transl_factor
    transl = (transl_x, transl_y)
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
    if exclude_flip:
        do_flip = False
    else:
        do_flip = random.random() <= 0.5

    return scale, rot, transl, color_scale, do_flip


def augmentation(img, bbox, data_split, exclude_flip=False, out_img_shape=(256, 256)):
    if data_split == 'training' or data_split == 'train':
        scale, rot, transl, color_scale, do_flip = get_aug_config(exclude_flip)
    else:
        scale, rot, transl, color_scale, do_flip = 1.0, 0.0, (0.0, 0.0), np.array([1, 1, 1]), False
    img, trans, inv_trans, trans_joint \
        = generate_patch_image(img, bbox, scale, rot, transl, do_flip, out_img_shape)
    img = np.clip(img * color_scale[None, None, :], 0, 255)

    return img, trans, inv_trans, rot, do_flip

def generate_patch_image(cvimg, bbox, scale, rot, transl, do_flip, out_shape):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, transl)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, transl,
                                        inv=True)
    trans_joint = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], 1.0, 0.0, transl)

    return img_patch, trans, inv_trans, trans_joint

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, transl, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment translation
    src_center[0] += transl[0]
    src_center[1] += transl[1]

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans


def flip_hand(img, kps_2d):
    h, w, _ = img.shape
    img = cv2.flip(img, 1)
    kps_2d[:, 0] = w - 1 - kps_2d[:, 0]
    return img, kps_2d

