# Density convolution
We propose our convolution for 2D and 3D images with density function.

## Use
Import the dConv2D and dConv3D functions. 

You can call the dConv2D function through:
>out2d = dConv2D(I2d, filters, density, kernel_size, padding_size, stride_size, bias)

where:

- I2d: 2D image, format (batch, channels_in, height, width)
- filters: trainable kernel filters, in the format given by conv.weight
- density: density function values
- bias: (optional): trainable bias, in the format given by conv.bias
- out: convolved 2D image

You can call the dConv3D function through:
>out3d = dConv3D(I3d, filters, density, kernel_size, padding_size, stride_size, bias)

where:

- I3d: 3D image, format (batch, channels_in, depth, height, width)
- filters: trainable kernel filters, in the format given by conv.weight
- density: density function values
- bias: (optional): trainable bias, in the format given by conv.bias
- out: convolved 3D image

## Density function values
We suggest to test our proposed density function values, i.e.,:

- *den = [0.5]* for 3 x 3 kernel
- *den = [0.3, 1.5]* for 5 x 5 kernel
- *den = [0.06, 1.2, 1.6]* for 7 x 7 kernel

However, optimal values could be different for your application and can be fine-tuned by the user.

These values can be applied for both 2D and 3D convolution with density function.

## Test file


## Requirements
Python, PyTorch
