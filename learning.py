import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


def dConv2D(I,f,den,ksize,psize,ssize,grp=1,bias=None):
    den = den
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
    grpBatchView = grpBatch.view(1, 1, channels_out, 1, 1, 1).expand(pYfY_1.size(0), pYfY_1.size(1), -1, 1, pYfY_1.size(4), pYfY_1.size(5)).to(pYfY_1.device)
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

class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1, bias=True, den=[]):
        super(CustomConv2d, self).__init__()
        
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.den = den
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.weight_dir = "weightFolder"
        self.groups = groups
 
        os.makedirs(self.weight_dir, exist_ok=True)
        
        weight_filename = f"{in_channels}_{out_channels}_{kernel_size}_{groups}.pt"
        weight_path = os.path.join(self.weight_dir, weight_filename)

        if os.path.exists(weight_path):
            self.weight = nn.Parameter(torch.load(weight_path))
        else:
            weight = (torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
            nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')
            torch.save(weight, weight_path)
            self.weight = nn.Parameter(weight) 
                
    def forward(self, x):
        return dConv2D(x, self.weight, self.den, self.kernel_size, self.padding, self.stride, self.groups, self.bias)

class dModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = CustomConv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, groups=1, den=[0.75])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x
        
dNet = dModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(dNet.parameters(), lr=0.01)
inp = torch.randn(2,1,64,64)
tar = torch.randn(2,1,64,64)
for i in range(10):
    out = dNet(inp)
    Loss = criterion(out, tar)
    optimizer.zero_grad()
    Loss.backward()
    optimizer.step()
    print('LOSS = ', Loss.item())
