import torch
import torch.nn as nn
from dConv import dConv2D
from dConv import dConv3D

# This test example shows the convolution with density functions for 2D and 3D images
# A density function with uniform values equal to 1 gives the same results of standard convolution

batch = 4
channels_in = 4
channels_out = 8
rows = 64
cols = 78
kernel_size = 3
padding_size = 1
stride_size = 1

I2d = torch.randn(batch, channels_in, rows, cols)
conv = nn.Conv2d(channels_in, channels_out, (kernel_size, kernel_size), stride=(stride_size, stride_size), padding=(padding_size,padding_size), bias=True)
f = conv.weight
bi = conv.bias
den = [1.0]

d_out2d = dConv2D(I2d,f,den,kernel_size,padding_size,stride_size,bi)
out2d = conv(I2d)
print('max abs error 2D:', (d_out2d - out2d).abs().max().item())


batch = 3
channels_in = 2
channels_out = 1
depth = 8
rows = 64
cols = 78
kernel_size = 5
padding_size = 2
stride_size = 2

I3d = torch.randn(batch, channels_in, depth, rows, cols)
conv = nn.Conv3d(channels_in, channels_out, (kernel_size, kernel_size, kernel_size), stride=(stride_size, stride_size, stride_size), padding=(padding_size,padding_size, padding_size), bias=False)
f = conv.weight
den = [1.0,1.0]

d_out3d = dConv3D(I3d,f,den,kernel_size,padding_size,stride_size)
out3d = conv(I3d)

print('max abs error 3D:', (d_out3d - out3d).abs().max().item())
