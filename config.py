import torch
import argparse

class Config:
    dataset = ''
    output_root = './output'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # network
    model = 'MultiViewNet'
    backbone = 'resnet50'
    num_stage = 1
    num_FMs = 8#32 # 8
    view_idx = [0,1,2,3,4,5,6,7]
    num_view = len(view_idx)
    feature_size = 64
    num_vert = 778 
    num_joint = 21
    num_params = 6 * 16 + 10 + 3  # one hand
    root_idx = 9
    # -------------
    checkpoint = None  
    finetune = False
    continue_train = False
    use_GTR = True
    use_2D_GT = True
    random_mask = True
    multi_view_finetune = False
    # -------------
    # training
    batch_size = 8
    lr = 3e-4
    total_epoch = 150
    input_img_shape = (256, 256)
    bbox_3d_size = 0.4
    num_worker = 16
    # -------------
    # use blur and occlusion augmetation
    aug = False
    # -------------
    save_epoch = [10, 20, 30, 40, 48, 49]# list(range(total_epoch))
    # save_epoch = [9, 10, 23, 24, 25, 26, 27, 28, 29, 30]
    eval_interval = 1
    print_iter = 100

def update_config(cfg):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='HanCo')
    parser.add_argument('--model', type=str, default='MultiViewNet')
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--use_GTR', default=True)
    parser.add_argument('--use_2D_GT', default=False)
    parser.add_argument('--random_mask', default=False)
    parser.add_argument('--stage', type=int, default=1, dest='num_stage')
    parser.add_argument('--epoch', type=int, dest='total_epoch', default=50)
    parser.add_argument('--continue', dest='continue_train', default=False)
    parser.add_argument('--finetune', default=False)
    parser.add_argument('--blur_aug', default=False, dest='aug')
    parser.add_argument('--checkpoint', type=str, default='')
    args = parser.parse_args()
    
    # update config
    # the dest name should be consistent with the config attribute name.
    for key, value in args.__dict__.items():
        if hasattr(cfg, key):
            print(f'Set {key} to {value}.')
            cfg.__setattr__(key, value)
        else:
            print('Add new arguments:', (key, value))

    pre = 'example'

    cfg.__setattr__('experiment_name', '{}'.format(pre) + '_view_{}'.format(cfg.num_view) + '_aug_{}'.format(cfg.aug) + '_use2dGT_{}'.format(cfg.use_2D_GT) + '_useGTR_{}'.format(cfg.use_GTR) + '_randMask_{}'.format(cfg.random_mask) + '_{}'.format(cfg.model) + '_{}'.format(cfg.backbone) + '_{}'.format(cfg.dataset)\
                          + '_Stage{}'.format(cfg.num_stage) + '_Batch{}'.format(cfg.batch_size) + '_lr{}'.format(cfg.lr)\
                          + '_Size{}'.format(cfg.input_img_shape[0]) + '_Epochs{}'.format(cfg.total_epoch))
    return cfg

cfg = update_config(Config())

