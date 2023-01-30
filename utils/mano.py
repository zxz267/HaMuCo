import numpy as np
from manopth.manolayer import ManoLayer
import torch

# def fix_shape(mano_layer):
#     if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
#         print('Fix shapedirs bug of MANO')
#         mano_layer['left'].shapedirs[:, 0, :] *= -1

def fix_shape(mano_layer):
    assert hasattr(mano_layer['left'], 'th_shapedirs') or hasattr(mano_layer['left'], 'shapedirs')
    if hasattr(mano_layer['left'], 'th_shapedirs'):
        if torch.sum(torch.abs(mano_layer['left'].th_shapedirs[:, 0, :] - mano_layer['right'].th_shapedirs[:, 0, :])) < 1:
            print('Fix shapedirs bug of MANO')
            mano_layer['left'].th_shapedirs[:, 0, :] *= -1
    else:
        if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
            print('Fix shapedirs bug of MANO')
            mano_layer['left'].shapedirs[:, 0, :] *= -1

class MANO(object):
    def __init__(self, side='right'):
        self.layer = self.get_layer(side)
        self.vertex_num = 778
        self.face = self.layer.th_faces.numpy()
        self.joint_regressor = self.layer.th_J_regressor.numpy()

        self.joint_num = 21
        self.template = self.layer.th_v_template.numpy()
        self.joints_name = ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinly_4')
        self.skeleton = ( (0,1), (0,5), (0,9), (0,13), (0,17), (1,2), (2,3), (3,4), (5,6), (6,7), (7,8), (9,10), (10,11), (11,12), (13,14), (14,15), (15,16), (17,18), (18,19), (19,20) )
        self.root_joint_idx = self.joints_name.index('Wrist')

        # add fingertips to joint_regressor
        self.fingertip_vertex_idx = [745, 317, 444, 556, 673] # mesh vertex idx (right hand)
        thumbtip_onehot = np.array([1 if i == 745 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        indextip_onehot = np.array([1 if i == 317 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        middletip_onehot = np.array([1 if i == 445 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        ringtip_onehot = np.array([1 if i == 556 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        pinkytip_onehot = np.array([1 if i == 673 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        self.joint_regressor = np.concatenate((self.joint_regressor, thumbtip_onehot, indextip_onehot, middletip_onehot, ringtip_onehot, pinkytip_onehot))
        self.joint_regressor = self.joint_regressor[[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],:]

    def get_layer(self, side):
        return ManoLayer(side=side, mano_root='./mano/models', use_pca=False, flat_hand_mean=False)  # load right hand MANO model


if __name__ == '__main__':
    mano = ManoLayer(flat_hand_mean=True)
    # Generate random shape parameters
    random_shape = torch.zeros(1, 10)
    # Generate random pose parameters, including 3 values for global axis-angle rotation
    random_pose = torch.zeros(1, 6 + 3)
    hand_verts, hand_joints = mano(random_pose, random_shape)
    scale = torch.norm(hand_joints[0, 10] - hand_joints[0, 9], p=2, dim=-1)
    print(scale)
