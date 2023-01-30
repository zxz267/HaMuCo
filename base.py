import os
import cv2
import numpy as np
import time
import torch

from data.dataset import get_dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from models.model import get_model
from config import cfg
from utils.logger import setup_logger
from data.processing import getSquare, batch_torch_uvd2xyz

class Trainer:
    def __init__(self):
        log_folder = os.path.join(cfg.output_root, 'log')
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        logfile = os.path.join(log_folder, 'train_' + cfg.experiment_name + '.log')
        self.logger = setup_logger(output=logfile, name="Training")
        self.logger.info('Start training: %s' % ('train_' + cfg.experiment_name))

    def get_optimizer(self, model):
        optimizer = optim.AdamW([{'params': model.parameters(), 'initial_lr': cfg.lr}], cfg.lr)
        self.logger.info('The parameters of the model are added to the optimizer.')
        return optimizer

    def get_schedule(self, optimizer):
        schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=cfg.total_epoch,
                                                        eta_min=0)
        self.logger.info('The learning rate schedule for the optimizer has been set.')
        return schedule

    def load_model(self, model, optimizer, schedule):
        checkpoint = torch.load(cfg.checkpoint)
        self.logger.info("Loading the model of epoch-{} from {}...".format(checkpoint['last_epoch'], cfg.checkpoint))
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        schedule.load_state_dict(checkpoint['schedule'])
        start_epoch = checkpoint['last_epoch'] + 1
        self.logger.info('The model is loaded successfully.')
        return start_epoch, model

    def load_model_finetune(self, model):
        checkpoint = torch.load(cfg.checkpoint)
        self.logger.info("Loading the model of epoch-{} from {}...".format(checkpoint['last_epoch'], cfg.checkpoint))
        save_model = checkpoint['net']
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        self.logger.info('The model is loaded successfully.')
        return model

    def save_model(self, model, optimizer, schedule, epoch):
        save = {
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'schedule': schedule.state_dict(),
            'last_epoch': epoch
        }
        path_checkpoint = os.path.join(cfg.output_root, 'checkpoint', cfg.experiment_name)
        if not os.path.exists(path_checkpoint):
            os.makedirs(path_checkpoint)
        save_path = os.path.join(path_checkpoint, "checkpoint_epoch[%d_%d].pth" % (epoch, cfg.total_epoch))
        torch.save(save, save_path)
        self.logger.info('Save checkpoint to {}'.format(save_path))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr

    def _make_batch_loader(self, shuffle=True, split='train', drop_last=True):
        self.logger.info("Creating dataset...")
        self.batch_loader = DataLoader(get_dataset(cfg.dataset, split),
                                       batch_size=cfg.batch_size,
                                       num_workers=cfg.num_worker,
                                       shuffle=shuffle,
                                       pin_memory=True,
                                       drop_last=drop_last)
        self.logger.info("The dataset is created successfully.")

    def _make_model(self, eval=False):
        self.logger.info("Making the model...")
        model = get_model(cfg.model).to(cfg.device)
        optimizer = self.get_optimizer(model)
        schedule = self.get_schedule(optimizer)
        if cfg.continue_train:
            start_epoch, model = self.load_model(model, optimizer, schedule)
        elif cfg.finetune:
            model = self.load_model_finetune(model)
            start_epoch = 0
        else:
            start_epoch = 0
        model.train()
        if eval:
            model.eval()
        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer
        self.schedule = schedule
        self.logger.info("The model is made successfully.")



