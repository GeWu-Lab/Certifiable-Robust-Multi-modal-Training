import json
import random
import os
import numpy as np
import torch
import torch.nn as nn
from models.multimodal import MultiModalModel
from models.backbone import resnet18
from train import *
from utils import *
import os
import hydra
from torch.utils.tensorboard import SummaryWriter

def build_model(cfg):
    if cfg.modality == 'None':
        video_model = hydra.utils.instantiate(cfg.encoder_v).to(cfg.device)
        audio_model = hydra.utils.instantiate(cfg.encoder_a).to(cfg.device)
        model = MultiModalModel(video_model, audio_model,cfg.n_classes,cfg.fusion_method)
        cfg.num_modal = 2
    print(cfg.method, cfg.fusion_method)
    return model

@hydra.main(config_path='cfgs', config_name='train', version_base=None)

def main(cfg):
    print(cfg)
    setup_seed(cfg.random_seed)
    model = build_model(cfg)
    if cfg.train:
        if cfg.tensorboard:
            tb_writer = SummaryWriter(log_dir=cfg.result_path)
        else:
            tb_writer = None

        (train_loader, val_loader) = get_dataset(cfg)
        train_logger, train_batch_logger, val_logger = get_logger(cfg)
        model = nn.DataParallel(model, device_ids=cfg.gpu_device).cuda()
        parameters = [p for p in model.parameters()]

        optimizer = hydra.utils.instantiate(cfg.optimizer, params = parameters)
        scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer = optimizer)

        train(train_loader,val_loader,model,train_logger,val_logger,train_batch_logger,tb_writer,cfg,\
                optimizer, scheduler)

# method=CRMT_JT,CRMT_AT,CRMT_mix

if __name__ == '__main__':
	main()

