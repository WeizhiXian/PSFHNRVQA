# -*- coding: utf-8 -*-

import os
import random
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchsummary import summary
from modules.dataset_for_block_npy import MyDataset, Preprocesser
from modules.model import MyModel, MyHuberLoss
from modules.utils import get_PLCC, get_RMSE, get_SROCC, mos_scatter, get_KROCC
from tensorboardX import SummaryWriter
from opts import parse_opts

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

show_detail = True


def val_model(my_model, device, criterion, val_loader, MOS_range):
    my_model.eval()
    with torch.no_grad():

        val_pred = {}
        val_label = {}
        val_epoch_loss = 0
        for i, inputs in enumerate(val_loader):

            frames = inputs['frames'].to(device)
            # res, map0, map1, map2, map3, map4 = my_model.mapout(frames)
            label = inputs['label'].to(device)
            file_name = inputs['file_name']

            output = my_model(frames)
            loss = criterion(output, label)  # 算损失

            val_epoch_loss += loss.item()
            l = label.view(-1).cpu().numpy()  # 一维array
            p = output[:, -1].cpu().detach().numpy()

            for j, name in enumerate(file_name):
                if name not in val_pred.keys():
                    val_pred[name] = []
                    val_label[name] = l[j]
                val_pred[name].append(p[j])

            if show_detail:
                print('\033[1;33m'
                      '----------------------------------------'
                      'Validating: %d / %d'
                      '----------------------------------------'
                      ' \033[0m' % (i + 1, len(val_loader)))
                print('\033[1;31m loss: ', loss.item(), '\033[0m')
 
        out = open('test_result.csv','w', newline='')
        csv_write = csv.writer(out,dialect='excel')
        
        video_val_pred = []
        video_val_label = []
        for d in val_pred.keys():
            video_val_label.append(val_label[d])
            video_val_pred.append(np.mean(val_pred[d]))
            csv_write.writerow([val_label[d], np.mean(val_pred[d])])

        video_val_label = np.array(video_val_label)
        video_val_pred = np.array(video_val_pred)

        val_rmse = get_RMSE(video_val_pred, video_val_label, MOS_range)
        val_plcc = get_PLCC(video_val_pred, video_val_label)
        val_srocc = get_SROCC(video_val_pred, video_val_label)
        val_krocc = get_KROCC(video_val_pred, video_val_label)
        val_loss = val_epoch_loss / (i + 1)

        csv_write.writerow(['loss', val_loss])
        csv_write.writerow(['rmse', val_rmse])
        csv_write.writerow(['plcc', val_plcc])
        csv_write.writerow(['srocc', val_srocc])
        csv_write.writerow(['krocc', val_krocc])
        
        print(val_loss, val_rmse, val_plcc, val_srocc, val_krocc)

        return val_loss, val_rmse, val_plcc, val_srocc, val_krocc

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':

    # 设置随机数种子
    setup_seed(20)

    opt = parse_opts()

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")

    video_dir = opt.video_dir
    score_file_path = opt.score_file_path
    info_file_path = opt.info_file_path
    MOS_range = [opt.MOS_min, opt.MOS_max]

    train_clip_stride = opt.train_clip_stride
    clip_stride = opt.clip_stride
    block_mode = opt.block_mode
    block_stride = opt.block_stride
    batch_size = opt.batch_size

    model_load_path = r'./model-save/final_model_LIVE_Qualcomm.pth'

    prep = Preprocesser(video_dir, score_file_path, info_file_path,
                        train_clip_stride, clip_stride, block_stride)

    val_dataset = MyDataset(prep, MOS_range=MOS_range, mode='val')

    print('load data')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    my_model = MyModel(device=device)

    criterion = MyHuberLoss()

    # 加载模型
    if not model_load_path:
        raise Exception('no model_load_path')
    print('load model:', model_load_path)
    model = torch.load(model_load_path)
    my_model.load_state_dict(model['net'])

    my_model.to(device)

    print('test model')
    val_model(my_model, device, criterion, val_loader, MOS_range)