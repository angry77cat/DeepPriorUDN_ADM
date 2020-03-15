import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torchvision import transforms
import scipy
import scipy.signal
import scipy.io as sio


class Set14Dataset(Dataset):
    def __init__(self, sigma):
        super(Set14Dataset, self).__init__()
        np.random.seed(9487)
        file_list = sorted(glob.glob('data/Set14/test-*'))
        
        self.xs = [] 
        self.ks = []
        self.ns = []
        for f in file_list:
            img = cv2.imread(f + '/gt.png', 0)
            img = torch.from_numpy(img).unsqueeze(0)
            self.xs.append(img)
            
            img = cv2.imread(f + '/kernel.png', 0)
            img = img/np.sum(img)
            img = img.astype(np.float32)
            img = np.expand_dims(img, axis=0)
            self.ks.append(img)
            
            img = sio.loadmat(f + '/noise.mat')['noise']
            self.ns.append(img)
        
        self.sigma = sigma
        self.im_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])
        
        self.file_list = file_list
        
    def __getitem__(self, index):
        '''
        self.xs: list, each uint, [ch, h, w]
        self.ks: np.array, float, [bs, ch, h, w]
        '''
        xs = self.xs[index]
        xs = self.im_transforms(xs) # float [ch, h, w]

        ks = self.ks[index] # float [ch, h, w]
        bs = scipy.signal.convolve2d(xs[0,...], ks[0,...], mode = 'same', boundary = 'wrap')
        bs = np.expand_dims(bs, axis = 0) + self.ns[index] * self.sigma / 255.0
        bs = bs.clip(0., 1.) # float [ch, h, w]
        fn = self.file_list[index].split('/')[-1].split('.')[0]
        return xs, ks, bs, fn

    def __len__(self):
        return len(self.xs)