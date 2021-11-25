# -*- coding: utf-8 -*-

import os
import torch
import time
import numpy as np
from eval import val_model
from modules.utils import get_PLCC, get_RMSE, get_SROCC, mos_scatter, get_KROCC, adjust_learning_rate

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

show_detail = True
interval = 1  # 控制validate和保存断点
adj_lr = True


def train_validate_model(my_model, device, criterion, optimizer, train_loader, val_loader,
                         start_epoch, epoch_nums, save_checkpoint, model_save_dir,
                         start_time, writer_t, writer_v, MOS_range):
    train_loss_list, train_plcc_list, train_rmse_list = [], [], []
    val_loss_list, val_plcc_list, val_rmse_list = [], [], []

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, start_epoch + epoch_nums):
        ################################################
        # 训练阶段 一次读取 batch_size 个数据
        my_model.train()
        # train_loader.dataset.set_idx(epoch)

        train_pred = {}
        train_label = {}
        train_epoch_loss = 0

        for i, inputs in enumerate(train_loader):

            optimizer.zero_grad()
            frames = inputs['frames'].to(device, non_blocking=True)
            # res, map0, map1, map2, map3, map4 = my_model.mapout(frames) 
            label = inputs['label'].to(device, non_blocking=True)
            file_name = inputs['file_name']

            with torch.cuda.amp.autocast():
                # model模型处理(n,c,f,h,w)格式的数据，n为batch-size
                output = my_model(frames)  # output: [[q0, q1, q2, q3, q]]
                loss = criterion(output, label)  # 算损失
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()  # 反向传播
            # optimizer.step()  # 更新参数

            # 记录
            train_epoch_loss += loss.item()
            l = label.view(-1).cpu().numpy()  # 一维array 长度为n
            p = output[:, -1].cpu().detach().numpy()  # 只要最后的预测q

            for j, name in enumerate(file_name):
                if name not in train_pred.keys():
                    train_pred[name] = []
                    train_label[name] = l[j]
                train_pred[name].append(p[j])

            if show_detail:
                print('\033[1;33m'
                      '----------------------------------------'
                      'Epoch: %d / %d    Done: %d / %d    %s'
                      '----------------------------------------'
                      ' \033[0m' % (epoch + 1, epoch_nums, i + 1, len(train_loader), start_time))
                if train_plcc_list:
                    print('\033[1;35m train_plcc_list:[', ' '.join(['%.4f' % x for x in train_plcc_list]), ']\033[0m')
                if val_plcc_list:
                    print('\033[1;35m val_plcc_list:  [', ' '.join(['%.4f' % x for x in val_plcc_list]), ']\033[0m')
                if train_loss_list:
                    print('\033[1;35m train_loss_list:[', ' '.join(['%.4f' % x for x in train_loss_list]), ']\033[0m')
                if val_loss_list:
                    print('\033[1;35m val_loss_list:  [', ' '.join(['%.4f' % x for x in val_loss_list]), ']\033[0m')

                print('\033[1;31m loss: ', loss.item(), '\033[0m')
                # print('\033[1;34m label: ', label.view(-1).data, '\033[0m')
                # print('\033[1;34m predict: ', output[:, -1].data, '\033[0m')
                # print('output: ', output)
            
            time.sleep(0.003)

        if adj_lr:
            # 调整学习率
            adjust_learning_rate(optimizer, epoch)

        video_train_pred = []
        video_train_label = []

        for d in train_pred.keys():
            video_train_pred.append(np.mean(train_pred[d]))
            video_train_label.append(train_label[d])

        video_train_pred = np.array(video_train_pred)
        video_train_label = np.array(video_train_label)

        train_rmse = get_RMSE(video_train_pred, video_train_label, MOS_range)
        train_plcc = get_PLCC(video_train_pred, video_train_label)
        train_srocc = get_SROCC(video_train_pred, video_train_label)
        train_krocc = get_KROCC(video_train_pred, video_train_label)

        train_loss_list.append(train_epoch_loss / (i + 1))
        train_plcc_list.append(train_plcc)
        train_rmse_list.append(train_rmse)

        writer_t.add_scalar('loss', train_epoch_loss / (i + 1), epoch)
        writer_t.add_scalar('rmse', train_rmse, epoch)
        writer_t.add_scalar('plcc', train_plcc, epoch)
        writer_t.add_scalar('srocc', train_srocc, epoch)
        writer_t.add_scalar('krocc', train_krocc, epoch)
        writer_t.add_scalar('learning rate', optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)
        writer_t.add_figure('Pred vs. MOS', mos_scatter(video_train_pred, video_train_label), epoch)
        # show_kernal(my_model, writer_t)

        # 保存断点
        if save_checkpoint:
            if epoch % interval == interval - 1:
                ckpt = {
                    'net': my_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }
                torch.save(ckpt, model_save_dir + '/checkpoint/ckpt_%s.pth' % (str(epoch)))

        ####################################################
        # 测试阶段
        if epoch % interval != interval - 1:
            val_loss_list.append(0)
            val_plcc_list.append(0)
            val_rmse_list.append(0)
            continue

        val_loss, val_rmse, val_plcc, val_srocc, val_krocc, fig = \
            val_model(my_model, device, criterion, val_loader, MOS_range)

        val_loss_list.append(val_loss)
        val_plcc_list.append(val_plcc)
        val_rmse_list.append(val_rmse)

        writer_v.add_scalar('loss', val_loss, epoch)
        writer_v.add_scalar('rmse', val_rmse, epoch)
        writer_v.add_scalar('plcc', val_plcc, epoch)
        writer_v.add_scalar('srocc', val_srocc, epoch)
        writer_v.add_scalar('krocc', val_krocc, epoch)
        writer_v.add_figure('Pred vs. MOS', fig, epoch)

    writer_t.close()
    writer_v.close()

    # 保存最后的model
    final_model = {
        'net': my_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': start_epoch + epoch_nums
    }
    torch.save(final_model, model_save_dir + '/final_model.pth')
