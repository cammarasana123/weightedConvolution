import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class dModel(nn.Module):
    def __init__(self):
        super(dModel, self).__init__()
        ksize = 3
        psize = 1
        ssize = 1
        channels_in = 1
        channels_out = 1
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=(ksize, ksize), stride=(ssize,ssize), padding=(psize,psize), bias=False )
        
    def dConv2D(I,f, wx, ksize,psize,ssize, bias=False):
        [batch_size,channels,rows,cols] = I.size()
        kh, kw = ksize, ksize
        dh, dw = ssize, ssize
    
        imageX = F.pad(I, (psize,psize,psize,psize), "constant",0)
        patches = imageX.unfold(2, kh, dh).unfold(3, kw, dw)
        patches = patches.contiguous().view(batch_size, channels, -1, kh, kw)
        patches = patches.permute(0, 2, 1, 3, 4)
        kx = torch.cat([torch.tensor(den),torch.tensor([1.0]),torch.flip(torch.tensor(den), dims=[0]) ])
        ky = kx.reshape(-1,1)
        patchesX = patches.unsqueeze(2)
        filtX = f.unsqueeze(0).unsqueeze(1)
        patchesX_filtX = patchesX * filtX
        pXfX_1 = torch.sum(patchesX_filtX,3)
        patchesX_filtX = pXfX_1.unsqueeze(3)
        patchesX_filt_weight1 = torch.matmul(kx,patchesX_filtX)
        patchesX_filt_weight2 = torch.matmul(patchesX_filt_weight1,ky)       
        if bias is False:
            res = patchesX_filt_weight2 
        else:
            bias = bias.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4)
            res = patchesX_filt_weight2 + bias    
        res = res.permute(0,4,3,2,1)    
        h_resize = int((rows-ksize+2*psize)/dh)+1
        w_resize = int((cols-ksize+2*psize)/dw)+1
        res = res.view(batch_size, -1, h_resize, w_resize)
        return res

    def forward(self, x, den, train_: bool=True, print_: bool=False, return_bottlenecks: bool=False):
        filt = self.conv.weight
        out = dModel.dConv2D(x, filt, den, 3, 1, 1)
        return out

dNet = dModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(dNet.parameters(), lr=0.01)
inp = torch.randn(2,1,64,64)
tar = torch.randn(2,1,64,64)
den = [0.5]
out = dNet(inp, den)
Loss = criterion(out, tar)
optimizer.zero_grad()
Loss.backward()
optimizer.step()

