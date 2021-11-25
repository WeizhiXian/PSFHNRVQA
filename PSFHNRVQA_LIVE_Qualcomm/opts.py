import argparse
import os
import time


KoNViD_1k_video_dir = r'F:/PSFHNRVQA/seq/KoNViD_1k/KoNViD_1k_videos'
KoNViD_1k_score_path = r'F:/PSFHNRVQA/seq/KoNViD_1k/KoNViD_1k_attributes.csv'
KoNViD_1k_split_info_path = r'./datas/KoNViD_1k_split.csv'

CVD2014_video_dir = r'F:/PSFHNRVQA/seq/CVD2014/CVD2014_videos'
CVD2014_score_path =r'F:/PSFHNRVQA/seq/CVD2014/CVD2014_attributes.csv'
CVD2014_split_info_path = r'./datas/CVD2014_split.csv'

LIVE_Qualcomm_video_dir = r'F:/PSFHNRVQA/seq/LIVE_Qualcomm/LIVE_Qualcomm_MP4'
LIVE_Qualcomm_score_path = r'F:/PSFHNRVQA/seq/LIVE_Qualcomm/LIVE_Qualcomm_attributes.csv'
LIVE_Qualcomm_split_info_path = r'./datas/LIVE_Qualcomm_split.csv'

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
INPUT_LENGTH = 8
INPUT_SIZE = 256

resnet_pth = os.path.join(PROJECT_PATH, 'pth/r3d18_KM_200ep.pth')


def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', default='train', type=str, help='train, retrain or predict')

    parser.add_argument('--video_dir', default=LIVE_Qualcomm_video_dir, type=str, help='Path to input videos')
    parser.add_argument('--score_file_path', default=LIVE_Qualcomm_score_path, type=str,
                        help='Path to input subjective score')
    parser.add_argument('--info_file_path', default=LIVE_Qualcomm_split_info_path, type=str,
                        help='Path to input subjective score')
    parser.add_argument('--MOS_min', default=1.22, type=float, help='MOS min range')
    parser.add_argument('--MOS_max', default=4.64, type=float, help='MOS max range')

    parser.add_argument('--train_clip_stride', default=64, type=int, help='stride between clips during training')
    parser.add_argument('--clip_stride', default=64, type=int, help='stride between clips')
    parser.add_argument('--block_mode', default=True, type=bool, help='use blocks')
    parser.add_argument('--block_stride', default=256, type=int, help='stride between random blocks during training')

    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=5e-8, type=float, help='L2 regularization')
    parser.add_argument('--epoch_nums', default=15, type=int, help='epochs to train')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')

    start_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    parser.add_argument('--start_time', default=start_time, type=str,
                        help='start time of this process')
    parser.add_argument('--save_model', default='./model-save/' + start_time, type=str,
                        help='path to save the model')
    parser.add_argument('--save_checkpoint', default=True, type=bool, help='')
    parser.add_argument('--load_model', default=r'./model-save/ckpt_0.pth', type=str,
                         help='path to load checkpoint')
    parser.add_argument('--writer_t_dir', default='./runs/' + start_time + '_train', type=str,
                        help='batch size to train')
    parser.add_argument('--writer_v_dir', default='./runs/' + start_time + '_val', type=str, help='batch size to train')

    args = parser.parse_args()

    if not os.path.isdir('./model-save/'):
        os.mkdir('./model-save/')
    if not os.path.isdir('./runs/'):
        os.mkdir('./runs/')

    return args


if __name__ == '__main__':
    args = parse_opts()
    print(args)
