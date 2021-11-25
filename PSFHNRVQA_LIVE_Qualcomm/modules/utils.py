import csv
import os
import random
import numpy as np
import cv2
import torch
import torch.nn.init as init
import torch.nn as nn
import torchvision
import matplotlib
from PIL import Image
from scipy import stats
from opts import INPUT_LENGTH, INPUT_SIZE
from modules.aug import get_normalize

matplotlib.use('Agg')
import matplotlib.pyplot as plt

normalize = get_normalize()

def adjust_learning_rate(optimizer, epoch):
    lr = optimizer.param_groups[0]['lr']

    if lr <= 1e-6:
        return
    step = 2
    if epoch % step == step - 1:
        cur_lr = lr * 0.5
        new_lr = cur_lr if cur_lr > 1e-6 else 1e-6
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def weigth_init(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d)):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def MOS_label(MOS, MOS_range):
    MOS_min, MOS_max = MOS_range
    # label = (MOS - MOS_min) / (MOS_max - MOS_min)
    # label = MOS # for KoNViD_1k
    # label = MOS/25+1 # for CVD2014
    label = (MOS-15)/15+1 # for LIVE_Qualcomm
    return label


def label_MOS(label, MOS_range):
    MOS_min, MOS_max = MOS_range
    # MOS = label * (MOS_max - MOS_min) + MOS_min
    # MOS = label # for KoNViD_1k
    # MOS = (label-1)*25 # for CVD2014
    MOS = (label-1)*15+15 # for LIVE_Qualcomm
    return MOS


def get_PLCC(y_pred, y_val):
    return stats.pearsonr(y_pred, y_val)[0]


def get_SROCC(y_pred, y_val):
    return stats.spearmanr(y_pred, y_val)[0]


def get_KROCC(y_pred, y_val):
    return stats.stats.kendalltau(y_pred, y_val)[0]


def get_RMSE(y_pred, y_val, MOS_range):
    y_p = label_MOS(y_pred, MOS_range)
    y_v = label_MOS(y_val, MOS_range)
    return np.sqrt(np.mean((y_p - y_v) ** 2))


def get_MSE(y_pred, y_val, MOS_range):
    y_p = label_MOS(y_pred, MOS_range)
    y_v = label_MOS(y_val, MOS_range)
    return np.mean((y_p - y_v) ** 2)


def mos_scatter(pred, mos, show_fig=False):
    fig = plt.figure()
    plt.scatter(mos, pred, s=5, c='g', alpha=0.5)
    plt.xlabel('MOS')
    plt.ylabel('PRED')
    plt.plot([0, 1], [0, 1], linewidth=0.5)
    if show_fig:
        plt.show()
    return fig


def fig2data(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data


def read_video_npy(video_path, start=None, intput_length=None):
    
    video_frames = np.load(video_path)
    if start and intput_length:
        video_frames = video_frames[start: start + intput_length]

    # 8*256*256*3 -> 256*256*3 分8个
    # 对每一个进行预处理，再堆叠

    img0, _ = normalize(video_frames[0,:,:,:], video_frames[0,:,:,:])
    img1, _ = normalize(video_frames[1,:,:,:], video_frames[1,:,:,:])
    img2, _ = normalize(video_frames[2,:,:,:], video_frames[2,:,:,:])
    img3, _ = normalize(video_frames[3,:,:,:], video_frames[3,:,:,:])
    img4, _ = normalize(video_frames[4,:,:,:], video_frames[4,:,:,:])
    img5, _ = normalize(video_frames[5,:,:,:], video_frames[5,:,:,:])
    img6, _ = normalize(video_frames[6,:,:,:], video_frames[6,:,:,:])
    img7, _ = normalize(video_frames[7,:,:,:], video_frames[7,:,:,:])

    img0 = np.expand_dims(img0,axis=0)
    img1 = np.expand_dims(img1,axis=0)
    img2 = np.expand_dims(img2,axis=0)
    img3 = np.expand_dims(img3,axis=0)
    img4 = np.expand_dims(img4,axis=0)
    img5 = np.expand_dims(img5,axis=0)
    img6 = np.expand_dims(img6,axis=0)
    img7 = np.expand_dims(img7,axis=0)

    video_frames = np.concatenate((img0,img1,img2,img3,img4,img5,img6,img7), axis=0)

    frames = torch.from_numpy(video_frames)
    frames = frames.permute([3, 0, 1, 2])  # 维度变换 (3, time, height, width)
    frames = frames.float()
    return frames


def save_crop_video(video_path, video_save_dir, clip_stride, n_clips):
    if not os.path.exists(video_save_dir):
        os.mkdir(video_save_dir)

    video_name = os.path.basename(video_path)

    cap = cv2.VideoCapture()
    cap.open(video_path)
    if not cap.isOpened():
        raise Exception("VideoCapture failed!")

    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frames_1 = np.zeros((length, H, W, 3), dtype='uint8')
    video_frames_2 = np.zeros((length, H, W, 3), dtype='uint8')
    video_frames_3 = np.zeros((length, W, H, 3), dtype='uint8')
    video_frames_4 = np.zeros((length, W, H, 3), dtype='uint8')

    for i in range(length):
        rval, frame = cap.read()  # h,w,c
        frame = Image.fromarray(frame)
        if rval:
            video_frames_1[i] = np.array(frame)
            video_frames_2[i] = np.array(frame.transpose(Image.FLIP_LEFT_RIGHT))
            video_frames_3[i] = np.array(frame.transpose(Image.ROTATE_90))
            video_frames_4[i] = np.array(frame.transpose(Image.ROTATE_270))
            
        else:
            raise Exception("VideoCapture failed!")
    cap.release()

    print(video_frames_1.size, length, H, W)

    for clip in range(n_clips):
        pos = int(clip * clip_stride)
        hIdxMax = H - INPUT_SIZE
        wIdxMax = W - INPUT_SIZE

        hIdx = [INPUT_SIZE * i for i in range(0, hIdxMax // INPUT_SIZE + 1)]
        wIdx = [INPUT_SIZE * i for i in range(0, wIdxMax // INPUT_SIZE + 1)]
        if hIdxMax % INPUT_SIZE != 0:
            hIdx.append(hIdxMax)
        if wIdxMax % INPUT_SIZE != 0:
            wIdx.append(wIdxMax)
        for h in hIdx:
            for w in wIdx:
                video_save_path = os.path.join(video_save_dir, video_name + '_{}_{}_{}_1.npy'.format(clip, h, w))
                np.save(video_save_path,
                        video_frames_1[pos:pos + INPUT_LENGTH,
                        h:h + INPUT_SIZE, w:w + INPUT_SIZE, :])
                video_save_path = os.path.join(video_save_dir, video_name + '_{}_{}_{}_2.npy'.format(clip, h, w))
                np.save(video_save_path,
                        video_frames_1[pos + 16:pos + 16 + INPUT_LENGTH,
                        h:h + INPUT_SIZE, w:w + INPUT_SIZE, :])
                video_save_path = os.path.join(video_save_dir, video_name + '_{}_{}_{}_3.npy'.format(clip, h, w))
                np.save(video_save_path,
                        video_frames_2[pos + 8:pos + 8 + INPUT_LENGTH,
                        h:h + INPUT_SIZE, w:w + INPUT_SIZE, :])
                video_save_path = os.path.join(video_save_dir, video_name + '_{}_{}_{}_4.npy'.format(clip, h, w))
                np.save(video_save_path,
                        video_frames_2[pos + 24:pos + 24 + INPUT_LENGTH,
                        h:h + INPUT_SIZE, w:w + INPUT_SIZE, :])
                video_save_path = os.path.join(video_save_dir, video_name + '_{}_{}_{}_5.npy'.format(clip, h, w))
                np.save(video_save_path,
                        video_frames_3[pos + 16:pos + 16 + INPUT_LENGTH,
                        w:w + INPUT_SIZE, h:h + INPUT_SIZE, :])
                video_save_path = os.path.join(video_save_dir, video_name + '_{}_{}_{}_6.npy'.format(clip, h, w))
                np.save(video_save_path,
                        video_frames_4[pos + 24:pos + 24 + INPUT_LENGTH,
                        w:w + INPUT_SIZE, h:h + INPUT_SIZE, :])
                

def get_video_name(file_name):
    video_name = os.path.basename(file_name)
    if not video_name.endswith('.npy'):
        return video_name
    else:
        return video_name.split('.mp4')[0] + '.mp4'


# 读取csv文件
def read_score(csv_path):
    dict = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        head_row = next(reader)
        for row in reader:
            dict[row[2]] = float(row[3])
    return dict


def read_split(csv_path):
    info_dict = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        head_row = next(reader)
        for row in reader:  # name, class, n_clips
            info_dict[row[0]] = [row[1], int(row[2])]
    return info_dict


def write_split(csv_path, infos, wtype):
    with open(csv_path, wtype, encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if wtype == 'w':
            writer.writerow(['file_name', 'class', 'n_clips', 'hNums', 'wNums'])
        for info in infos:
            writer.writerow(info)


def count_clip(video_path, stride):
    vc = cv2.VideoCapture()
    vc.open(video_path)
    if not vc.isOpened():
        raise Exception("VideoCapture failed!")
    n_frames = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    tIdMax = n_frames
    n_clips = tIdMax // stride

    return int(n_clips)


def count_block(video_path, block_stride):
    vc = cv2.VideoCapture()
    vc.open(video_path)
    if not vc.isOpened():
        raise Exception("VideoCapture failed!")
    H = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
    W = vc.get(cv2.CAP_PROP_FRAME_WIDTH)

    hIdxMax = H - INPUT_SIZE
    wIdxMax = W - INPUT_SIZE
    hNums = hIdxMax // block_stride + 1 if hIdxMax % block_stride == 0 \
        else hIdxMax // block_stride + 2
    wNums = wIdxMax // block_stride + 1 if wIdxMax % block_stride == 0 \
        else wIdxMax // block_stride + 2

    return int(hNums), int(wNums)


def data_split(full_list, ratio, shuffle=True):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为3个子列表
    :param full_list: 数据列表
    :param ratio:     []
    :param shuffle:
    :return:
    """
    nums_total = len(full_list)
    offset1 = int(nums_total * ratio[0])
    offset2 = int(nums_total * (ratio[0] + ratio[1]))

    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset1]
    sublist_2 = full_list[offset1:offset2]
    sublist_3 = full_list[offset2:]
    return sublist_1, sublist_2, sublist_3
