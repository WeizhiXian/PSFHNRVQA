# -*- coding: utf-8 -*-

import os
import random
import torch
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torchsummary import summary
from modules.dataset_for_block_npy import MyDataset, Preprocesser
from modules.model import MyModel, MyHuberLoss
from tensorboardX import SummaryWriter
from modules.utils import weigth_init
from opts import parse_opts
from train import train_validate_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True
    
    # 设置随机数种子
    setup_seed(20)

    opt = parse_opts()

    model_type = opt.model_type

    video_dir = opt.video_dir
    score_file_path = opt.score_file_path
    info_file_path = opt.info_file_path
    MOS_range = [opt.MOS_min, opt.MOS_max]

    train_clip_stride = opt.train_clip_stride
    clip_stride = opt.clip_stride
    block_mode = opt.block_mode
    block_stride = opt.block_stride

    learning_rate = opt.learning_rate
    weight_decay = opt.weight_decay
    epoch_nums = opt.epoch_nums
    batch_size = opt.batch_size

    start_time = opt.start_time
    model_save_dir = opt.save_model
    model_load_path = opt.load_model
    writer_t_dir = opt.writer_t_dir
    writer_v_dir = opt.writer_v_dir
    save_checkpoint = opt.save_checkpoint

    opt = parse_opts()

    # Only for the first training, the model needs preprocessing (preprocess is set to True)
    prep = Preprocesser(opt.video_dir, opt.score_file_path, opt.info_file_path,
                       opt.train_clip_stride, opt.clip_stride,
                       ratio=[0.8, 0.2, 0], preprocess=True)


    prep = Preprocesser(video_dir, score_file_path, info_file_path,
                        train_clip_stride, clip_stride, block_stride)

    train_dataset = MyDataset(prep, MOS_range=MOS_range, mode='train')
    val_dataset = MyDataset(prep, MOS_range=MOS_range, mode='val')
    

    print('load data')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 损失
    criterion = MyHuberLoss()

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")

    my_model = MyModel(device=device)
    my_model.apply(weigth_init)


    optimizer = optim.Adam(filter(lambda p: p.requires_grad, my_model.parameters()),
                           lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)

    if model_type == 'retrain':
        # 加载模型
        if not model_load_path:
            raise Exception('no model_load_path')
        print('load model:', model_load_path)
        model = torch.load(model_load_path)  # 加载断点
        my_model.load_state_dict(model['net'])  # 加载模型可学习参数
        optimizer.load_state_dict(model['optimizer'])  # 加载优化器参数
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_epoch = model['epoch']
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, my_model.parameters()),
                           lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
    else:
        start_epoch = 0

    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)
    if not os.path.isdir(model_save_dir + '/checkpoint'):
        os.mkdir(model_save_dir + '/checkpoint')

    my_model.to(device)
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])
    # model['rep'].to(device)
    # model = model.module

    summary(my_model, input_size=[(3, 8, 256, 256)])

    writer_t = SummaryWriter(writer_t_dir, comment='MyNet')
    writer_v = SummaryWriter(writer_v_dir, comment='MyNet')

    print('train model')

    train_validate_model(my_model, device, criterion, optimizer, train_loader, val_loader,
                         start_epoch, epoch_nums, save_checkpoint, model_save_dir,
                         start_time, writer_t, writer_v, MOS_range)
