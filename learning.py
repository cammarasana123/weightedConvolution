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
        grp = 1
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=(ksize, ksize), stride=(ssize,ssize), padding=(psize,psize), groups=grp, bias=False )
        
    def dConv2D(I,f, wx, ksize,psize,ssize,grp,bias=None):
        [batch_size,channels_in,rows,cols] = I.size()
        channels_out = f.size(0)
        kh, kw = ksize, ksize
        dh, dw = ssize, ssize
        grp_ratio_in = int(channels_in / grp)
        grp_ratio_out = int(channels_out / grp)
    
        imageX = F.pad(I, (psize,psize,psize,psize), "constant",0)
        patches = imageX.unfold(2, kh, dh).unfold(3, kw, dw)
        patches = patches.contiguous().view(batch_size, channels_in, -1, kh, kw)
        patches = patches.permute(0, 2, 1, 3, 4)
        kx = torch.cat([torch.tensor(den, device=I.device), torch.tensor([1.0], device=I.device), torch.flip(torch.tensor(den, device=I.device), dims=[0])])
        ky = kx.reshape(-1,1)
        patchesX = patches.unsqueeze(2)
        patchesY = patchesX.view(patchesX.size(0),patchesX.size(1),patchesX.size(2),grp,grp_ratio_in,patchesX.size(4),patchesX.size(5))
        filtX = f.unsqueeze(0).unsqueeze(1)
        filtY = filtX.unsqueeze(3)
        patchesY_filtY = patchesY * filtY
        pYfY_1 = torch.sum(patchesY_filtY,4)
        grpBatch = (torch.arange(grp)).repeat_interleave(grp_ratio_out)
        grpBatchView = grpBatch.view(1, 1, channels_out, 1, 1, 1).expand(pYfY_1.size(0), pYfY_1.size(1), -1, 1, pYfY_1.size(4), pYfY_1.size(5))
        pYfY_2 = pYfY_1.gather(dim=3, index=grpBatchView)
        patchesY_filt_weight1 = torch.matmul(kx,pYfY_2)
        patchesY_filt_weight2 = torch.matmul(patchesY_filt_weight1,ky)      
        if bias is None:
            res = patchesY_filt_weight2 
        else:
            bias = bias.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4)
            res = patchesY_filt_weight2 + bias    
        res = res.permute(0,4,3,2,1)    
        h_resize = int((rows-ksize+2*psize)/dh)+1
        w_resize = int((cols-ksize+2*psize)/dw)+1
        res = res.view(batch_size, -1, h_resize, w_resize)
        return res

    def forward(self, x, den, train_: bool=True, print_: bool=False, return_bottlenecks: bool=False):
        filt = self.conv.weight
        out = dModel.dConv2D(x, filt, den, 3, 1, 1, 1, None)
        return out

dNet = dModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(dNet.parameters(), lr=0.01)
inp = torch.randn(2,1,64,64)
tar = torch.randn(2,1,64,64)
den = [1.0]
for i in range(10):
    out = dNet(inp, den)
    Loss = criterion(out, tar)
    optimizer.zero_grad()
    Loss.backward()
    optimizer.step()
    print('LOSS = ', Loss.item())

