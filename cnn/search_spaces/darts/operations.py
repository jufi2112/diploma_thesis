# Defines operations that can get utilized during cell search
# Observations: 
#   - ReLU gets applied before Convolutional layers, not afterwards (see e.g. ReLUConvBN class)
#       while, if applied to all layers, the end result should not differ from doing BN(ReLU(Conv)), it still 
#       leaves open the question as to why this is done?

import torch
import torch.nn as nn

OPS = {
    "none": lambda C, stride, affine: Zero(stride),
    "avg_pool_3x3": lambda C, stride, affine: nn.AvgPool2d(
        3, stride=stride, padding=1, count_include_pad=False
    ),
    "max_pool_3x3": lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    "skip_connect": lambda C, stride, affine: Identity()
    if stride == 1
    else FactorizedReduce(C, C, affine=affine),
    "sep_conv_3x3": lambda C, stride, affine: SepConv(
        C, C, 3, stride, 1, affine=affine
    ),
    "sep_conv_5x5": lambda C, stride, affine: SepConv(
        C, C, 5, stride, 2, affine=affine
    ),
    "sep_conv_7x7": lambda C, stride, affine: SepConv(
        C, C, 7, stride, 3, affine=affine
    ),
    "dil_conv_3x3": lambda C, stride, affine: DilConv(
        C, C, 3, stride, 2, 2, affine=affine
    ),
    "dil_conv_5x5": lambda C, stride, affine: DilConv(
        C, C, 5, stride, 4, 2, affine=affine
    ),
    "conv_7x1_1x7": lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine),
    ),
}


class ReLUConvBN(nn.Module):
    """Class that executes a ReLU followed by a 2D convolution and a batch normalization layer

    Args:
        C_in (int): Number of input channels to the convolutional layer
        C_out (int): The number of output channels the convolutional layer should result in, i.e. number of kernels of
            the convolutional layer.
        kernel_size (int): Size of the convolutional kernel applied to the input.
        stride (int or tuple of int): Stride for the convolutional layer.
        padding (int): Padding for the convolutional layer
        affine (bool): Affine parameter for the batch normalization.
            True means the BN will have learnable affine parameters.
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False
            ),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    """Class that implements a dilated convolution.
    The input x will be transformed as follows: BN(Conv(DilConv(ReLU(x)))), where the outer convolution has kernel size
    1 and is responsible for transforming the channel size to the requested C_out value, since the dilated 
    convolution keeps the input number of channels C_in.

    Args:
        C_in (int): Input number of channels.
        C_out (int): Output number of channels, see Note section below.
        kernel_size (int): Kernel size for the dilated convolution
        stride (int or tuple of int): Stride for the dilated convolution
        padding (int): Padding for the dilated convolution
        dilation (int): Delation factor for the dilated convolution.
        affine (bool): Affine parameter for the batch normalization.
            True means the BN will have learnable affine parameters.

    Note:
        @see https://towardsdatascience.com/understanding-2d-dilated-convolution-operation-with-examples-in-numpy-and-tensorflow-with-d376b3972b25
        The groups parameter of the Conv2d layer conrols how many input channels each convolution uses. 
            groups = C_in will cause each convolution to only utilize one input channel and will therefore result in 
            C_in output channels. 
            Because of this, C_in is also utilized for the out_channels parameter of the dilated convolution and the
            requested number of output channels is created by applying a second convolution with kernel size 1 and
            out_channels = C_out. 
            @see https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            @see https://stackoverflow.com/questions/46536971/how-to-use-groups-parameter-in-pytorch-conv2d-function
    """
    def __init__(
        self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True
    ):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    """Class that implements a separable convolution.
    Internally, the input x gets passed through two groups of BN(Conv(Conv(ReLU(x)))), TODO: not sure why this is done
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    """Implements identity mapping, which simply returns the input"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    """Implements zero mapping by multiplying the input with 0"""
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)


class FactorizedReduce(nn.Module):
    """TODO
    """
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)  # why remove the first row and column?
        # Result of conv_1 and conv_2 have the same shape, possibly because of the stride and 0 padding
        out = self.bn(out)
        return out
