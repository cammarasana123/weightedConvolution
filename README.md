# Density convolution
We propose our convolution with density function for 2D and 3D images.

## Use
Import the dConv2D and dConv3D functions from the dConv.py file. 

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
We suggest to start your tests with the proposed density function values:

- *den = [0.5]* for 3 x 3 kernel
- *den = [0.3, 1.5]* for 5 x 5 kernel
- *den = [0.06, 1.2, 1.6]* for 7 x 7 kernel

These values can be applied for both 2D and 3D convolution with density function.

**Optimal values could be different for your application and can be fine-tuned by the user.**

You can check our optimal values on our reference paper. For examples, the optimal density function for low-complexity problems and 3 x 3 kernels is *den = [1.5]*



## Test files
The *single test* file allows the user to test the convolution with density function and compare the results with torch.nn convolution. We underline that
the results are the same when the density function values are equal to 1.

The *learning* file implements a minimal learning model applying the convolution with density function.

## Requirements
Python, PyTorch
