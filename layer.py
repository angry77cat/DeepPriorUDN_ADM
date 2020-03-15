import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EdgeDetect(nn.Module):
    def __init__(self,  \
                 k_size = 9, \
                 sig_list = [0.70710678, 1.58113883, 3.16227766, 7.07106781]):
        super(EdgeDetect, self).__init__()
        self.filters = [self._gen_filter(k_size, scale) for scale in sig_list]

    def forward(self, x, thre):
        with torch.no_grad():
            grads = []
            for filt in self.filters:
                fx = torch.from_numpy(filt[0]).float().unsqueeze(0).unsqueeze(0).to(device)
                fy = torch.from_numpy(filt[1]).float().unsqueeze(0).unsqueeze(0).to(device)
                ix = self._t_conv2d(x, fx)
                iy = self._t_conv2d(x, fy)
                grad = (ix ** 2 + iy ** 2) ** 0.5
                grads.append(grad)
            grads = torch.stack(grads)
            grads = torch.max(grads, 0).values
            bin_grads = self._binary(grads, thre= thre)
            return bin_grads
        
    def _gen_filter(self, l, sig):
        x, y = np.meshgrid(np.linspace(-int(l/2), int(l/2), l), np.linspace(-int(l/2), int(l/2), l))
        gauss1 = np.sign(x) * np.exp(-(x**2+y**2)/2/sig**2)
        gauss1 /= np.sum(np.abs(gauss1))
        gauss2 = np.sign(y) * np.exp(-(x**2+y**2)/2/sig**2)
        gauss2 /= np.sum(np.abs(gauss2))
        return (gauss1, gauss2)

    def _t_conv2d(self, i, fx):
        '''
        i: [bs, ch, h, w]
        fx: [bs, ch, h, w]
        '''
        h, w = fx.shape[2], fx.shape[3]
        h_pad, w_pad = int(h/2), int(w/2)
        pad_i = F.pad(i, pad = (w_pad,w_pad,h_pad,h_pad), mode='circular')

        h_slice = -1 if h % 2 == 0 else None
        w_slice = -1 if w % 2 == 0 else None
        return F.conv2d(pad_i, fx)[:,:,:h_slice,:w_slice]
    
    def _binary(self, x, thre = 0.05):
        w = torch.exp(x) - 1
        w[w > thre] = 1
        w[w <= thre] = 0
        return w

class ISubLayer(nn.Module):
    def __init__(self):
        super(ISubLayer, self).__init__()

    def forward(self, z, y, k, rho):
        '''
        z: from Z-sub problem [bs, ch, h, w]
        y: blurred image [bs, ch, h, w]
        k: kernel [bs, 1, h, w]
        '''
        bs, ch, h, w = z.shape
        
        denominators = []
        upperlefts = []
        for i in range(bs):
            V = self._t_psf2otf(k[i,0,...], [h, w]) # [1, h, w, 2]
            denominator = self._t_abs(V) ** 2 # [1, h, w]
            
            upperleft = self._t_multiply(self._t_conj(V), \
                                   torch.rfft(y[i, 0, ...].unsqueeze(0), signal_ndim = 2, onesided = False)\
                                  ) # [1, h, w, 2]
            denominators.append(denominator)
            upperlefts.append(upperleft)           
        
        z_s = []
        for i in range(bs):
            z_ = self._t_c2r_divide(upperlefts[i] + \
                           rho * torch.rfft(z[i, 0, ...].unsqueeze(0), signal_ndim = 2, onesided = False)\
                           , denominators[i] + rho) # [1, h, w, 2]
            z_ = torch.irfft(z_, signal_ndim = 2, onesided = False) # [1, h, w]
            z_s.append(z_)
        z_s = torch.stack(z_s, dim = 0)
        return z_s
            
    def _t_psf2otf(self, psf, shape):
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

    def _t_abs(self, input):
        '''
        @input: [bs, h, w, 2]
        @output: [bs, h, w]
        '''
        r, i = input[:,:,:,0], input[:,:,:,1]
        return (r ** 2 + i ** 2) ** 0.5

    def _t_conj(self, input):
        '''
        @input: [bs, h, w, 2]
        @output: [bs, h, w, 2]
        '''
        input_ = input.clone()
        input_[:,:,:,1] = -input_[:,:,:,1]
        return input_

    def _t_multiply(self, t1, t2):
        '''
        @input: [bs, h, w, 2]
        @output: [bs, h, w, 2]
        '''
        real1, imag1 = t1[:,:,:,0], t1[:,:,:,1]
        real2, imag2 = t2[:,:,:,0], t2[:,:,:,1]
        return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim = -1)
   
    def _t_c2r_divide(self, t1, t2):
        '''
        complex divided by real
        @input: [bs, h, w, 2], [bs, h, w]
        @output: [bs, h, w, 2]
        '''
        real1, imag1 = t1[..., 0], t1[..., 1]
        return torch.stack([real1/t2, imag1/t2], dim = -1)
    
class AdmEdgeDetect(nn.Module):
    def __init__(self,  \
                 k_size = 9, \
                 sig_list = [0.70710678, 1.58113883, 3.16227766, 7.07106781]):
        super(AdmEdgeDetect, self).__init__()
        self.filters = [self._gen_filter(k_size, scale) for scale in sig_list]

    def forward(self, x, base, idx, ite, u_thre, l_thre):
        with torch.no_grad():
            grads = []
            for filt in self.filters:
                fx = torch.from_numpy(filt[0]).float().unsqueeze(0).unsqueeze(0).to(device)
                fy = torch.from_numpy(filt[1]).float().unsqueeze(0).unsqueeze(0).to(device)
                ix = self._t_conv2d(x, fx)
                iy = self._t_conv2d(x, fy)
                grad = (ix ** 2 + iy ** 2) ** 0.5
                grads.append(grad)
            grads = torch.stack(grads)
            grads = torch.max(grads, 0).values
            bin_grads = self._binaryv2(grads, base, idx, ite, u_thre, l_thre)
            return grads, bin_grads
        
    def _gen_filter(self, l, sig):
        x, y = np.meshgrid(np.linspace(-int(l/2), int(l/2), l), np.linspace(-int(l/2), int(l/2), l))
        gauss1 = np.sign(x) * np.exp(-(x**2+y**2)/2/sig**2)
        gauss1 /= np.sum(np.abs(gauss1))
        gauss2 = np.sign(y) * np.exp(-(x**2+y**2)/2/sig**2)
        gauss2 /= np.sum(np.abs(gauss2))
        return (gauss1, gauss2)

    def _t_conv2d(self, i, fx):
        '''
        i: [bs, ch, h, w]
        fx: [bs, ch, h, w]
        '''
        h, w = fx.shape[2], fx.shape[3]
        h_pad, w_pad = int(h/2), int(w/2)
        pad_i = F.pad(i, pad = (w_pad,w_pad,h_pad,h_pad), mode='circular')

        h_slice = -1 if h % 2 == 0 else None
        w_slice = -1 if w % 2 == 0 else None
        return F.conv2d(pad_i, fx)[:,:,:h_slice,:w_slice]
    
    def _binaryv2(self, x, base, idx, ite, u_thre = 0.05, l_thre = 0.05):
        w = torch.pow(base, x) - 1
        w[w > u_thre + 0.0 * (idx/ite)] = 1
        w[w < l_thre + 0.0 * (idx/ite)] = 0
        return w
