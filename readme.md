# A Pyramidal Spatiotemporal Feature Hierarchy Based No-Reference Video Quality Assessment via Transfer Learning

## Environments

* python == 3.6.13
* pytorch == 1.8.0
* torchvision == 0.9.0
* torchsummary==1.5.1
* matplotlib == 3.3.2
* scipy == 1.5.2
* opencv_python == 4.4.0
* numpy == 1.19.5
* tensorboardX == 2.1

## Command statements

### Train the Model
```cd /d F:\PSFHNRVQA\PSFHNRVQA_LIVE_Qualcomm```

```python F:\PSFHNRVQA\PSFHNRVQA_LIVE_Qualcomm\start.py```

### Test the Model
```cd /d F:\PSFHNRVQA\PSFHNRVQA_LIVE_Qualcomm```

```python F:\PSFHNRVQA\PSFHNRVQA_LIVE_Qualcomm\test.py```

### Check Training log (Tensorboard visualization)
```python -m tensorboard.main --logdir=F:\PSFHNRVQA\PSFHNRVQA_LIVE_Qualcomm\runs --host localhost```

### YUV2MP4
* When trainning on LIVE_Qualcomm dataset, all videos should be transferred in to 'mp4' format by ffmpeg.
* The ffmpeg command is 'yuv2mp4.cmd' in F:\PSFHNRVQA\seq\LIVE_Qualcomm
* All mp4 videos should be put in F:\PSFHNRVQA\seq\LIVE_Qualcomm\LIVE_Qualcomm_MP4

## Pre-trained models

The Pre-trained of models trained on LIVE_Qualcomm, CVD2014, and KoNViD-1k are put in  <a href="">https://pan.baidu.com/s/17Q3eBmeBncSsIgF-QKkQ4g</a> (password: FSFH).


## Architecture

![General Architecture](https://github.com/WeizhiXian/PSFHNRVQA/blob/master/general_architecture.png)

