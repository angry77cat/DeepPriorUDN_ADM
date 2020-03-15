import torch
import torch.nn as nn
import torch.nn.init as init
from layer import ISubLayer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
class UDN(nn.Module):
    def __init__(self, rho, n_layer):
        super(UDN, self).__init__()
        self.solver = ISubLayer()
        for i in range(n_layer):
            self.add_module('zsub-{:02d}'.format(i), UDN_Z_Module().to(device))
        self.rho = rho
        self.n_layer = n_layer
        
    def forward(self, b, k):
        Z = b
        output = []
        for i in range(self.n_layer):
            I = self.solver(Z, b, k, rho = self.rho[i])
            Z = self._modules['zsub-{:02d}'.format(i)](I)[0]
            output.append(Z)
        return output
    
class UDN_Z_Module(nn.Module):
    def __init__(self):
        super(UDN_Z_Module, self).__init__()
        kernel_size = 3
        padding = 1
        n_channels = 64

        self.add_module('layer0', nn.Sequential(nn.Conv2d(in_channels = 1, 
                                                        out_channels = n_channels, 
                                                        kernel_size = kernel_size, 
                                                        stride = 1,
                                                        padding = 1,
                                                        dilation = 1,
                                                        bias = True), 
                                              nn.ReLU(inplace=True)))
        
        for idx, dila in enumerate([2,3,4,3,2]):
            self.add_module('layer{:d}'.format(idx+1), nn.Sequential(nn.Conv2d(in_channels = n_channels, 
                                                        out_channels = n_channels, 
                                                        kernel_size = kernel_size, 
                                                        stride = 1,
                                                        padding = dila,
                                                        dilation = dila,
                                                        bias = True), 
                                                                     nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95),
                                              nn.ReLU(inplace=True)))

        self.add_module('layer6', nn.Sequential(nn.Conv2d(in_channels = n_channels, 
                                                        out_channels = 1, 
                                                        kernel_size = kernel_size, 
                                                        stride = 1,
                                                        padding = 1,
                                                        dilation = 1,
                                                        bias = True), 
                                              ))

        self._initialize_weights()
        
    def forward(self, x, gt = False):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2) 
        x4 = self.layer4(x3)
        x5 = self.layer5(x4 + x2)
        x6 = self.layer6(x5 + x1)
        return x6 + x, 0
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
                
class ADM(nn.Module):
    def __init__(self):
        super(ADM, self).__init__()
        ds = []
        for i in range(1,26):
            m = ADM_Z_Module()
            path = 'models/ADM_Z/model_{:02d}_iter050.pth'.format(i)
            m.load_state_dict(torch.load(path)['stat_dict'])
            m = m.to(device).eval()
            ds.append(m)
        self.denoiser = ds

    def forward(self, x, noise_level = None):
        '''
        noise_level: torch.LongTensor (bs)
        '''
        bs, ch, h, w = x.shape
        with torch.no_grad():
            denoised = self.denoiser[noise_level.numpy()[0]](x)

        return denoised
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
                
class ADM_Z_Module(nn.Module):
    def __init__(self):
        super(ADM_Z_Module, self).__init__()
        kernel_size = 3
        padding = 1
        n_channels = 64

        layers = []
        conv = nn.Conv2d(in_channels = 1, \
                         out_channels = n_channels, \
                         kernel_size = kernel_size, \
                         stride = 1, \
                         padding = 1, \
                         dilation = 1, \
                         bias = True)
        layers.append(conv)
        layers.append(nn.ReLU(inplace=True))
        for _ in [2,3,4,3,2]:
            conv = nn.Conv2d(in_channels = n_channels, \
                             out_channels = n_channels, \
                             kernel_size = kernel_size, \
                             stride = 1, \
                             padding = _, \
                             dilation = _, \
                             bias = True)
            layers.append(conv)
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        conv = nn.Conv2d(in_channels = n_channels, \
                         out_channels = 1, \
                         kernel_size = kernel_size, \
                         stride = 1, \
                         padding = 1, \
                         dilation = 1, \
                         bias = True)
        layers.append(conv)
        self.comp1 = nn.Sequential(*layers)
        self._initialize_weights()
            
        
    def forward(self, x):
        res = self.comp1(x)
        return x - res
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)