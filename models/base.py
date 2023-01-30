import torch
import torch.nn as nn
from torchvision.transforms.functional import normalize
from config import cfg
import numpy as np
from abc import abstractmethod

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def process_single_view_input(self, x):
        img = x['img']
        height, width = img.shape[-2], img.shape[-1]
        assert height == cfg.input_img_shape[0] and width == cfg.input_img_shape[1], 'Please check the input image shape!'
        x = img.view(-1, 3, height, width)

        # for list input
        if isinstance(x, list):
            x = np.stack(x, axis=0)
            x = torch.FloatTensor(x)
            x = x.to(cfg.device)
            batch, h, w, c = x.shape
            # input normalize
            x = x.permute(0, 3, 1, 2).reshape(batch, c, h, w)
            x = normalize(x / 255, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            # register view number
            self.view_num = batch
            return x
        else:
            assert len(x.shape) == 3 or len(x.shape) == 4, x.shape
            # The dataloader input has already been normalized.
            if self.training or (len(x.shape) == 4 and x.shape[0] != 1):
                return x.to(cfg.device)
            else:
                # for RGB raw image input test
                if isinstance(x, np.ndarray):
                    x = torch.FloatTensor(x)
                x = x.to(cfg.device)
                # if not batch inputs
                if len(x.shape) == 3:
                    x = x.unsqueeze(0)
                batch, h, w, c = x.shape
                # input normalize
                x = x.permute(0, 3, 1, 2).reshape(batch, c, h, w)
                x = normalize(x / 255, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                return x

    @abstractmethod
    def compute_loss(self, outputs, targets):
        pass

    @abstractmethod
    def process_output(self, outputs):
        pass

