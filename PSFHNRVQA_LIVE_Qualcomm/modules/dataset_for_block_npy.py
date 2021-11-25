import os
import torch
from torch.utils.data import Dataset
from modules.utils import get_video_name, write_split, count_clip, read_split, data_split, read_score, MOS_label, \
    save_crop_video, read_video_npy


# 读取已经分块后的npy
# train:all
# val:all

class Preprocesser():
    def __init__(self, video_dir, score_path, info_path, train_clip_stride, clip_stride,
                 ratio=None, preprocess=False):

        print("\033[1;34m video_dir: ", video_dir, "\033[0m")

        self.train_clip_stride = train_clip_stride
        self.clip_stride = clip_stride
        self.video_paths = [x.path for x in os.scandir(video_dir) if
                            x.name.endswith('.mp4')]
        self.label_dict = read_score(score_path)

        self.video_train_dir = os.path.join(os.path.dirname(video_dir), 'save_train')
        self.video_val_dir = os.path.join(os.path.dirname(video_dir), 'save_val')
        self.video_test_dir = os.path.join(os.path.dirname(video_dir), 'save_test')

        if preprocess:
            if not ratio:
                raise Exception("Set ratio!")
            self.train_paths, self.val_paths, self.test_paths = \
                data_split(self.video_paths, ratio, shuffle=True)
            self.write_infos(info_path)
            self.info_dict = read_split(info_path)
            self.crop_video_blocks()
        else:
            self.train_paths, self.val_paths, self.test_paths = [], [], []
            self.info_dict = read_split(info_path)
            for vp in self.video_paths:
                vn = get_video_name(vp)
                if self.info_dict[vn][0] == 'train':
                    self.train_paths.append(vp)
                elif self.info_dict[vn][0] == 'val':
                    self.val_paths.append(vp)
                elif self.info_dict[vn][0] == 'test':
                    self.test_paths.append(vp)

    def write_infos(self, path):
        infos = []
        for vp in self.train_paths:
            vn = get_video_name(vp)
            infos.append([vn, 'train', str(count_clip(vp, self.train_clip_stride))])
        for vp in self.val_paths:
            vn = get_video_name(vp)
            infos.append([vn, 'val', str(count_clip(vp, self.clip_stride))])
        for vp in self.test_paths:
            vn = get_video_name(vp)
            infos.append([vn, 'test', str(count_clip(vp, self.clip_stride))])
        write_split(path, infos, 'w')

    def crop_video_blocks(self):

        for i, vp in enumerate(self.video_paths):
            vn = get_video_name(vp)
            print(i, vn)
            if self.info_dict[vn][0] == 'train':
                video_save_dir = self.video_train_dir
                clip_stride = self.train_clip_stride
                mode = 'train'
            elif self.info_dict[vn][0] == 'val':
                video_save_dir = self.video_val_dir
                clip_stride = self.clip_stride
                mode = 'val'
            else:
                video_save_dir = self.video_test_dir
                clip_stride = self.clip_stride
                mode = 'test'
            n_clips = self.info_dict[vn][1]
            save_crop_video(vp, video_save_dir, clip_stride, n_clips)


class MyDataset(Dataset):

    def __init__(self, prep, MOS_range, mode):

        self.label_dict = prep.label_dict

        self.MOS_range = MOS_range
        self.mode = mode

        if self.mode == 'train':
            video_dir = prep.video_train_dir
        elif self.mode == 'val':
            video_dir = prep.video_val_dir
        else:
            video_dir = prep.video_test_dir

        self.video_paths = [x.path for x in os.scandir(video_dir) if
                            x.name.endswith('.npy')]

    def __getitem__(self, index):
        file_path = self.video_paths[index]
        frames = read_video_npy(file_path)

        file_name = get_video_name(file_path)
        label = torch.tensor(self.label_dict[file_name]).view(-1)
        label = MOS_label(label, self.MOS_range)

        sample = {
            'file_path': file_path,
            'file_name': file_name,
            'frames': frames,
            'label': label
        }

        return sample

    def __len__(self):
        return len(self.video_paths)
