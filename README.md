# UPDATE

The weighted convolution has been updated to version 2.0. Please refer to:

>https://github.com/cammarasana123/weightedConvolution2.0

# Weighted Convolution
We propose our convolution with density function for 2D and 3D images.

## Use
Import the dConv2D and dConv3D functions from the dConv.py file. 

You can call the dConv2D function through:
>out2d = dConv2D(I2d, filters, density, kernel_size, padding_size, stride_size, groups, bias)

where:

- I2d: 2D image, format (batch, channels_in, height, width)
- filters: trainable kernel filters, in the format given by conv.weight
- density: density function values
- bias: (optional): trainable bias, in the format given by conv.bias
- out: convolved 2D image

You can call the dConv3D function through:
>out3d = dConv3D(I3d, filters, density, kernel_size, padding_size, stride_size, groups, bias)

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

**Optimal values could be different for your application and can be fine-tuned by the user. They shall be treated as hyper-parameters. **

You can check our optimal values on the reference paper.



## Test files
The *single test* file allows the user to test the convolution with density function and compare the results with torch.nn convolution. We underline that
the results are the same when the density function values are equal to 1.

The *learning* file implements a minimal learning model applying the weighted convolution.

## Requirements
Python, PyTorch

Tested with: Python 3.11.10, PyTorch 2.6.0

## Requirements
Please refer to the following articles:

Simone Cammarasana and Giuseppe Patanè. Optimal Density Functions for Weighted Convolution in Learning Models. 2025. DOI: https://arxiv.org/abs/2505.24527.

Simone Cammarasana and Giuseppe Patanè. Optimal Weighted Convolution for Classification and Denosing. 2025. DOI: https://arxiv.org/abs/2505.24558.


