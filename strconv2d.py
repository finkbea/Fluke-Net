# Caelan Booker
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair

# ===============================================================================================
# StrengthConv2d
# ===============================================================================================

class StrengthConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', strength_flag=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(StrengthConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        # These steps need to happen after the super call, because we need
        # some params to be set so we can modify them        
        self.strength_flag = strength_flag

        # If the strength flag is set, we need to disable gradient calculating on
        # the weights, and create a new parameter which will serve as our strength
        # tensor. I also create a few test tensors to use for comparing how the
        # weights and strengths change across calls.
        # The strength tensor is shaped a bit funny, but trust me, it needs to have
        # that NCHW pytorch format in order for each strength to multiply its
        # corresponding kernel properly. I also initialize the strengths to 1.
        if self.strength_flag :
            self.weight.requires_grad=False
            self.strength = Parameter(torch.ones(out_channels,1,1,1))
            self.test_wei = self.weight.data.clone()
            self.test_str = torch.zeros(out_channels,1,1,1)
        
    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input, debug_flags=(False, False, False, False)):
        # If our strength flag is set, we need to multiply all the kernels by
        # their respective weights before convolution.
        if self.strength_flag :
            
            if debug_flags[0] :
                print(self.strength)
            if debug_flags[1] :
                print(self.weight)
            if debug_flags[2] :
                print(torch.all(self.test_str.eq(self.strength)))
            if debug_flags[3] :
                print(torch.all(self.test_wei.eq(self.weight)))
                
            output = self._conv_forward(input, self.strength * self.weight)

            if debug_flags[0] or debug_flags[2] :
                self.test_str = self.strength.data.clone()
                
            return output
        else :
            return self._conv_forward(input, self.weight)
