import torch
import torch.nn.functional as F

def dConv2D(I,f,den,ksize,psize,ssize,bias=False):
    
    # --- INPUT
    ## I: 2D image, format (batch, channels_in, height, width)
    ## f: kernel filters, in the format given by conv.weight
    ## den: density function values
    ## ksize: kernel size
    ## psize: padding size
    ## ssize: stride size
    ## bias (optional): bias, in the format given by conv.bias
    
    # --- OUTPUT
    ## res: convolved 2D image
    
    [batch_size,channels,rows,cols] = I.size()
    kh, kw = ksize, ksize
    dh, dw = ssize, ssize

    imageX = F.pad(I, (psize,psize,psize,psize), "constant",0)
    patches = imageX.unfold(2, kh, dh).unfold(3, kw, dw)
    patches = patches.contiguous().view(batch_size, channels, -1, kh, kw)
    patches = patches.permute(0, 2, 1, 3, 4)
    kx = torch.cat([torch.tensor(den, device=I.device), torch.tensor([1.0], device=I.device), torch.flip(torch.tensor(den, device=I.device), dims=[0])])
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

def dConv3D(I,f,den,ksize,psize,ssize,bias=False):
    
    # --- INPUT
    ## I: volumetric image, format (batch, channels_in, depth, height, width)
    ## f: kernel filters, in the format given by conv.weight
    ## den: density function values
    ## ksize: kernel size
    ## psize: padding size
    ## ssize: stride size
    ## bias (optional): bias, in the format given by conv.bias
    
    # --- OUTPUT
    ## res: convolved 3D image
    
    [batch_size,channels,depth,rows,cols] = I.size()
    kh, kw, kd = ksize, ksize, ksize
    dh, dw, dd = ssize, ssize, ssize 

    imageX = F.pad(I, (psize,psize,psize,psize,psize,psize), "constant",0)
    patches = imageX.unfold(2, kh, dh).unfold(3, kw, dw).unfold(4,kd,dd)
    patches = patches.contiguous().view(batch_size, channels, -1, kh, kw, kd)
    patches = patches.permute(0, 2, 1, 3, 4, 5)
    kx = torch.cat([torch.tensor(den, device=I.device), torch.tensor([1.0], device=I.device), torch.flip(torch.tensor(den, device=I.device), dims=[0])])
    ky = kx.reshape(-1,1)
    kz = kx.view(1,1,1,1,1,ksize)
    patches2 = patches.unsqueeze(2)
    filt2 = f.unsqueeze(0).unsqueeze(1)    
    patches2_filt2 = patches2 * filt2
    p2f2_1 = torch.sum(patches2_filt2,3) 
    p2f2_2 = p2f2_1.unsqueeze(3)
    patches2_filt_weight1 = torch.matmul(kx,p2f2_2)
    patches2_filt_weight2 = torch.matmul(patches2_filt_weight1,ky)
    patches2_filt_weight3 = torch.tensordot(patches2_filt_weight2,kz,dims = ([4,5],[5,4]))
    patches2_filt_weight3 = patches2_filt_weight3.squeeze(7).squeeze(6)
    if bias is False:
        res = patches2_filt_weight3 
    else:
        bias = bias.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4).unsqueeze(5)
        res = patches2_filt_weight3 + bias            
    res = res.permute(0,5,4,3,2, 1)    
    h_resize = int((rows-ksize+2*psize)/dh)+1
    w_resize = int((cols-ksize+2*psize)/dw)+1
    d_resize = int((depth-ksize+2*psize)/dd)+1
    res = res.view(batch_size, -1, d_resize, h_resize, w_resize)
    return res
