import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
import scipy
import scipy.signal
import matplotlib.pyplot as plt
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F
from skimage import filters
import nvgpu
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def readimg(path):
    img = cv2.imread(path, 0)
    img = torch.from_numpy(img).unsqueeze(0)
    tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])
    return tf(img).unsqueeze(0)

def t_psf2otf(psf, shape):
    '''
    otf: [1, h, w, 2]
    '''
    imshape_ = psf.shape
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(imshape_, dtype=int)
    dshape = shape - imshape
    # padding
    idx, idy = np.indices(imshape)
    offx, offy = 0, 0
    pad_psf = torch.zeros(list(shape), dtype = psf.dtype).to(device)
    pad_psf[idx + offx, idy + offy] = psf
    for axis, axis_size in enumerate(imshape_):
        pad_psf = torch.roll(pad_psf, -int(axis_size / 2), dims = axis)
    otf = torch.rfft(pad_psf.unsqueeze(0), signal_ndim = 2, onesided = False)
    return otf

def t_multiply(t1, t2):
    '''
    @input: [bs, h, w, 2]
    @output: [bs, h, w, 2]
    '''
    real1, imag1 = t1[:,:,:,0], t1[:,:,:,1]
    real2, imag2 = t2[:,:,:,0], t2[:,:,:,1]
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim = -1)

class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    print(file_list)
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)
    
def zero_pad(image, shape, position='corner'):
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueErro ("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img

def psf2otf(psf, shape):
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)
    return otf


def save_image(data, cm, fn):
   
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.imshow(data, cmap=cm)
    
    fig.add_axes(ax)
    plt.title('hi')
#     fig.set_size_inches(width/height, 1, forward=True)
#     plt.savefig(fn, dpi = height) 
    plt.savefig(fn) 
    plt.close()
    
def ratio_determine(sigma):
    sigma_rng = [2.55, 25]
    ratio0 = [9e-1, 6e-1]
    ratio1 = [9e-1, 3e-1]
    
    shift = (sigma - sigma_rng[0])/(sigma_rng[1] - sigma_rng[0])
    output = [\
        ratio0[0] + (ratio0[1] - ratio0[0]) * shift,\
        ratio1[0] + (ratio1[1] - ratio1[0]) * shift,\
    ]
    return output