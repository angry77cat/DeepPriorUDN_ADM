import time
import torch
import torch.nn as nn
import numpy as np
from layer import AdmEdgeDetect, ISubLayer, EdgeDetect
from iqa import calc_psnr_tensor, calc_ssim_tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def ADM_Deblur(model, sigma_, task_name, data, modelSigma1, modelSigma2, ite, early, base, u_thre, l_thre, ratio, verbose = True):
    record = {'init':[] , 'gt': [], 'I-sub':[], 'Z-sub':[], 'w':[], 'psnr':[], 'ssim':[], 'mse': []}
    modelSigmaS = np.logspace(np.log10(modelSigma1),np.log10(modelSigma2),ite)
    ns = np.ceil(modelSigmaS/2)
    
    edge = AdmEdgeDetect()
    solver = ISubLayer()
    mse = nn.MSELoss()
    
    start_time0 = time.time()
    with torch.no_grad():
        x, k, b, fn = data

        b = b.float().to(device)
        x = x.to(device)
        k = k.to(device)
        
        i = b
        record['init'].append(b[0,0,...].cpu().numpy())
        record['gt'].append(x[0,0,...].cpu().numpy())
        record['psnr'].append(calc_psnr_tensor(b.to(device), x))
        record['ssim'].append(calc_ssim_tensor(b.to(device), x))

        nriqa_input = []
        
        _ ,w = edge(i, base, 0, ite, u_thre, l_thre)
        
        ratio = np.logspace(np.log10(ratio[0]), np.log10(ratio[1]), ite)

        for idx in range(ite):
            if idx == early:
                break
            
            # Adaptive Deconv
            rho = (sigma_**2)/(modelSigmaS[idx]**2)
            i1 = solver(i, b, k, rho = rho)
            i2 = solver(i, b, k, rho = rho * ratio[idx])  
            i3 = i2 * w + i1 * (1 - w)
            # z sub problem
            idn = int(ns[idx])-1
            i_out = model(i3, torch.LongTensor([idn]))

            record['I-sub'].append(i3[0,0,...].cpu().numpy())
            record['Z-sub'].append(i_out[0,0,...].cpu().numpy())
            record['w'].append(w)
            record['psnr'].append(calc_psnr_tensor(i_out, x))
            record['ssim'].append(calc_ssim_tensor(i_out, x))
            record['mse'].append(mse(i_out, x))
            
            i = i_out
        
        if verbose:
            print('ADM deblur elapse time: {:2.4f}sec'.format(time.time() - start_time0))

            print('task: {:s} blurry: {:4.2f}/{:4.3f} restored: {:4.2f}/{:4.3f}'.format(
                  fn[0].split('/')[-1],\
                  calc_psnr_tensor(b.to(device), x),\
                  calc_ssim_tensor(b.to(device), x),\
                  calc_psnr_tensor(i_out, x),\
                  calc_ssim_tensor(i_out, x)))
    return record


def UDN_Deblur(model, data, ratio, n_layer, verbose = True):
    record = {'init':[] , 'gt': [], 'I-sub':[], 'Z-sub':[], 'psnr':[], 'ssim':[]}
    
    model.eval()
    edge = EdgeDetect()
    isolver = ISubLayer()
    start_time0 = time.time()
    with torch.no_grad():
        x, k, b, fn = data
        b = b.float().to(device)            
        x = x.to(device)
        k = k.float().to(device)

        z = b
        record['init'].append(b[0,0,...].cpu().numpy())
        record['gt'].append(x[0,0,...].cpu().numpy())
        record['psnr'].append(calc_psnr_tensor(b.to(device), x))
        record['ssim'].append(calc_ssim_tensor(b.to(device), x))
        
        ratio = np.logspace(np.log10(ratio[0]), np.log10(ratio[1]), n_layer)
        
        for i in range(n_layer):
            ia = isolver(z, b, k, rho = model.rho[i] * 1.)
            ib = isolver(z, b, k, rho = model.rho[i] * ratio[i])
            
            w = edge(z, thre = 0.2)
            i_ = ib * w + ia * (1 - w)
            z = model.__dict__['_modules']['zsub-{:02d}'.format(i)](i_)[0]
            record['psnr'].append(calc_psnr_tensor(z.to(device), x))
            record['ssim'].append(calc_ssim_tensor(z.to(device), x))
            record['Z-sub'].append(z[0,0,...].cpu().numpy())
            record['I-sub'].append(i_[0,0,...].cpu().numpy())

        if verbose:
            print('UDN deblur telapse time: {:2.4f}sec'.format(time.time() - start_time0))

            print('task: {:s} blurry: {:4.2f}/{:4.3f} restored: {:4.2f}/{:4.3f}'.format(
                  fn[0].split('/')[-1],\
                  calc_psnr_tensor(b.to(device), x),\
                  calc_ssim_tensor(b.to(device), x),\
                  calc_psnr_tensor(z, x),\
                  calc_ssim_tensor(z, x)))
    return record

def ADM_UDN_Combine(data, coarse_part, fine_part, verbose = True):
    h, w = coarse_part.shape
    x, k, b, fn = data
    x = x.to(device)
    b = b.float().to(device)  
    
    start_time0 = time.time()
    fine = torch.rfft(torch.from_numpy(fine_part).unsqueeze(0).unsqueeze(0), signal_ndim = 2, onesided = False).to(device)
    coarse = torch.rfft(torch.from_numpy(coarse_part).unsqueeze(0).unsqueeze(0), signal_ndim = 2, onesided = False).to(device)
    
    x_, y_ = np.meshgrid(np.linspace(-int(w/2), int(w/2), w), np.linspace(-int(h/2), int(h/2), h))
    grid = (x_**2 + y_**2)**0.5
    grid = (grid - np.min(grid)) / (np.max(grid) - np.min(grid))
    grid = 1/(1 + np.exp(-grid+0.5))
    grid = torch.FloatTensor(grid.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    grid = torch.stack([grid, grid], dim = -1)

    f1 = fine * (grid) + coarse * (1-grid)
    fb = torch.irfft(f1, signal_ndim = 2, onesided = False)
    
    if verbose:
        print('ADM_UDN deblur elapse time: {:2.4f}sec'.format(time.time() - start_time0))

        print('task: {:s} blurry: {:4.2f}/{:4.3f} restored: {:4.2f}/{:4.3f}'.format(
              fn[0].split('/')[-1],\
              calc_psnr_tensor(b.to(device), x),\
              calc_ssim_tensor(b.to(device), x),\
              calc_psnr_tensor(fb, x),\
              calc_ssim_tensor(fb, x)))
    return fb[0, 0,...].cpu().numpy()
