import torch
import torch.nn.functional as F

def dConv2D(I,f,den,ksize,psize,ssize,grp=1,bias=None):
    
    # --- INPUT
    ## I: 2D image, format (batch, channels_in, height, width)
    ## f: kernel filters, in the format given by conv.weight
    ## den: density function values
    ## ksize: kernel size
    ## psize: padding size
    ## ssize: stride size
    ## grp (optional): number of groups (default: 1)
    ## bias (optional): bias, in the format given by conv.bias (default: None)
    
    # --- OUTPUT
    ## res: convolved 2D image
    
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

def dConv3D(I,f,den,ksize,psize,ssize,grp=1,bias=None):
    
    # --- INPUT
    ## I: volumetric image, format (batch, channels_in, depth, height, width)
    ## f: kernel filters, in the format given by conv.weight
    ## den: density function values
    ## ksize: kernel size
    ## psize: padding size
    ## ssize: stride size
    ## grp (optional): number of groups (Default: 1)  
    ## bias (optional): bias, in the format given by conv.bias (default: None)
    
    # --- OUTPUT
    ## res: convolved 3D image
    
    [batch_size,channels_in,depth,rows,cols] = I.size()
    channels_out = f.size(0)
    kh, kw, kd = ksize, ksize, ksize
    dh, dw, dd = ssize, ssize, ssize 
    grp_ratio_in = int(channels_in / grp)
    grp_ratio_out = int(channels_out / grp)    

    imageX = F.pad(I, (psize,psize,psize,psize,psize,psize), "constant",0)
    patches = imageX.unfold(2, kh, dh).unfold(3, kw, dw).unfold(4,kd,dd)
    patches = patches.contiguous().view(batch_size, channels_in, -1, kh, kw, kd)
    patches = patches.permute(0, 2, 1, 3, 4, 5)
    kx = torch.cat([torch.tensor(den, device=I.device), torch.tensor([1.0], device=I.device), torch.flip(torch.tensor(den, device=I.device), dims=[0])])
    ky = kx.reshape(-1,1)
    kz = kx.view(1,1,1,1,1,ksize)
    patchesX = patches.unsqueeze(2)
    patchesY = patchesX.view(patchesX.size(0),patchesX.size(1),patchesX.size(2),grp,grp_ratio_in,patchesX.size(4),patchesX.size(5), patchesX.size(6) )
    filtX = f.unsqueeze(0).unsqueeze(1)    
    filtY = filtX.unsqueeze(3)
    patchesY_filtY = patchesY * filtY
    pYfY_1 = torch.sum(patchesY_filtY,4)
    grpBatch = (torch.arange(grp)).repeat_interleave(grp_ratio_out)
    grpBatchView = grpBatch.view(1, 1, channels_out, 1, 1, 1, 1).expand(pYfY_1.size(0), pYfY_1.size(1), -1, 1, pYfY_1.size(4), pYfY_1.size(5), pYfY_1.size(6))
    pYfY_2 = pYfY_1.gather(dim=3, index=grpBatchView)
    patchesY_filt_weight1 = torch.matmul(kx,pYfY_2)
    patchesY_filt_weight2 = torch.matmul(patchesY_filt_weight1,ky)
    patchesY_filt_weight3 = torch.tensordot(patchesY_filt_weight2,kz,dims = ([4,5],[5,4]))
    patchesY_filt_weight3 = patchesY_filt_weight3.squeeze(7).squeeze(6)
    if bias is None:
        res = patchesY_filt_weight3 
    else:
        bias = bias.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4).unsqueeze(5)
        res = patchesY_filt_weight3 + bias            
    res = res.permute(0,5,4,3,2, 1)    
    h_resize = int((rows-ksize+2*psize)/dh)+1
    w_resize = int((cols-ksize+2*psize)/dw)+1
    d_resize = int((depth-ksize+2*psize)/dd)+1
    res = res.view(batch_size, -1, d_resize, h_resize, w_resize)
    return res
