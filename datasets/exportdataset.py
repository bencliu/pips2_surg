from numpy import random
from numpy.core.numeric import full
import torch
import numpy as np
import os
import scipy.ndimage
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
from torch._C import dtype, set_flush_denormal
import utils.geom
import glob
import cv2
from pathlib import Path
import sys
import albumentations as A
from functools import partial

def augment_video(augmenter, **kwargs):
    assert isinstance(augmenter, A.ReplayCompose)
    keys = kwargs.keys()
    for i in range(len(next(iter(kwargs.values())))):
        data = augmenter(**{
            key: kwargs[key][i] if key not in ['bboxes', 'keypoints'] else [kwargs[key][i]] for key in keys
        })
        if i == 0:
            augmenter = partial(A.ReplayCompose.replay, data['replay'])
        for key in keys:
            if key == 'bboxes':
                kwargs[key][i] = np.array(data[key]).reshape(4)
            elif key == 'keypoints':
                kwargs[key][i] = np.array(data[key]).reshape(2)
            else:
                kwargs[key][i] = data[key]

def read_mp4(fn):
    vidcap = cv2.VideoCapture(fn)
    frames = []
    while(vidcap.isOpened()):
        ret, frame = vidcap.read()
        if ret == False:
            break
        frames.append(frame)
    vidcap.release()
    return frames

class ExportDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_location='./pod_export',
                 dataset_version='aa',
                 S=36,
                 N=21, #15
                 crop_size=(384,512), 
                 use_augs=False,
                 sample=None,
    ):
        print('loading export...')

        self.dataset_location = dataset_location
        self.S = S
        self.N = N
        self.H, self.W = crop_size
        self.crop_size = crop_size
        self.use_augs = use_augs
        
        if "pod_export" in dataset_location:
            self.dataset_location = Path('%s/%s' % (self.dataset_location, dataset_version))
        else:
            self.dataset_location = Path(dataset_location)

        folder_names = self.dataset_location.glob('*/')
        folder_names = [fn for fn in folder_names]
        folder_names = sorted(list(folder_names))
        print('found %d folders in %s' % (len(folder_names), self.dataset_location))

        self.all_folder_names = []
        rgbs = read_mp4(str(folder_names[0] / 'rgb.mp4'))
        rgbs = np.stack(rgbs, axis=0) # S,H,W,3
        S_local,H,W,C = rgbs.shape 
        if not H==self.H or not W==self.W:
            start_H = (H - self.H) // 2
            start_W = (W - self.W) // 2
            rgbs = rgbs[:, start_H:start_H+self.H, start_W:start_W+self.W, :]
            S_local,H,W,C = rgbs.shape 
        assert(H==self.H)
        assert(W==self.W)
        assert(S_local==self.S)
        
        if sample is not None:
            self.all_folder_names = self.filter_image_folders(folder_names[:sample])
        else:
            self.all_folder_names = self.filter_image_folders(folder_names)

        self.color_augmenter = A.ReplayCompose([
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.2),
            A.RGBShift(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.ImageCompression(quality_lower=50, quality_upper=100, p=0.3),
        ], p=0.8)

    #Get rid of videos that are too short 
    def filter_image_folders(self, folder_names): 
        new_folder_names = [] 
        for folder_name in folder_names:
            rgbs = read_mp4(str(folder_name / 'rgb.mp4'))
            d = dict(np.load(folder_name / 'track.npz', allow_pickle=True))
            track = d['track_g']
            if len(rgbs) == len(track):
                new_folder_names.append(folder_name) 
        return new_folder_names
        
    
    def __getitem__(self, index):
        folder = self.all_folder_names[index]
        
        rgbs = read_mp4(str(folder / 'rgb.mp4'))

        if len(rgbs)==0:
            print('corrupted mp4 in %s; returning fake' % folder)
            fake_sample = {
                'rgbs': np.zeros((self.S,3,self.H,self.W), dtype=np.uint8), 
                'track_g': np.zeros((self.S,self.N,4), dtype=np.float32) #Seq_len, number of keypoints, annotations per frame 
            }
            return fake_sample
        rgbs = np.stack(rgbs, axis=0) # S,H,W,3

        d = dict(np.load(folder / 'track.npz', allow_pickle=True))
        try:
            track = d['track_g']
        except Exception as e:
            print(d.keys())
            print(d)
            print(folder) 

        H,W,C = rgbs[0].shape
        S,N,D = track.shape #(30 seq_len, 21 keypoints, 4 [xy-vis-valid])

        assert(N >= self.N)
        assert(H==self.H)
        assert(W==self.W)
        assert(S==self.S)
        assert(D==4) # xy, vis, valid
        assert(C==3)
        
        if N > self.N:
            inds = np.random.choice(N, self.N, replace=False)
            track = track[:,inds]
        
        if self.use_augs:
            augment_video(self.color_augmenter, image=rgbs)

        rgbs = rgbs.transpose(0,3,1,2)
        rgbs = rgbs[:,::-1].copy() # BGR->RGB

        return {
            'rgbs': rgbs,
            'track_g': track,
        }

    def __len__(self):
        return len(self.all_folder_names)
