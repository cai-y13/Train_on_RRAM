import torch
import torch.nn as nn
from torch.autograd import Function, Variable
import math

OLD_PYTORCH = '1.12' in torch.__version__


def get_rram_param_groups(named_params):
    param_groups = []
    param_group = dict()
    bn_param_group = {'params': [], 'config': 'bn'}
    for n, p in named_params:
        if len(p.size()) > 1:  # conv / fc
            param_group['params'] = [p]
            param_group['config'] = 'conv' if len(p.size()) == 4 else 'fc'
            param_groups.append(param_group)
            param_group = dict()
        elif 'bn' in n:  # bn
            bn_param_group['params'].append(p)
        else:  # bias
            param_groups[-1]['params'].append(p)
    if len(bn_param_group['params']) > 0:
        param_groups.append(bn_param_group)
    return param_groups


def linear_fixed_function(tensors, fixed_bits):
    max_value, min_value = 0, float('inf')
    for tensor in tensors:
        max_value = max(torch.max(tensor.abs()), max_value)
        min_value = min(torch.min(tensor.abs()), min_value)
    step = (max_value - min_value) / (2 ** fixed_bits - 1)
    #step = max_value.
    if step > 0:
        for tensor in tensors:
            tensor.clamp_(min_value, max_value).sub_(min_value)
            tensor.div_(step).round_().mul_(step).add_(min_value)


def hardware_fixed_function(*tensors, fixed_bits, max_value=float('inf'), running_mode=True):
    max_num = 0
    if running_mode == True:
        for tensor in tensors:
            if max_num < torch.max(tensor.abs()).item():
                max_num = torch.max(tensor.abs()).item()
    else:
        max_num = max_value
    for tensor in tensors:
        #If input a max value, then use the defined scale. Otherwise, use the max value of the tensor.
        max_num = max(max_num, pow(2, -32))
        scale = pow(2, round(math.log2(max_num)))
        #if max_value > 1:
        step = 1 / (2 ** fixed_bits)
        tensor.div_(scale)
        tensor.clamp_(-1, 1).sub_(-1)
        tensor.div_(step).round_().mul_(step).add_(-1).mul_(scale)


class LinearFixedFunction(Function):
    """docstring for LinearFixedInputFunction in pytorch 0.1.12"""
    def __init__(self, fixed_bits, max_values={'forward':1, 'backward':10}, running_mode=True):
        super(LinearFixedFunction, self).__init__()
        self.fixed_bits = fixed_bits
        self.max_values = max_values
        self.running_mode = running_mode

    def forward(self, *input):
        hardware_fixed_function(*input, fixed_bits=self.fixed_bits['forward'],
                                max_value=self.max_values['forward'], running_mode=self.running_mode)
        return input

    def backward(self, *grad_output):
        hardware_fixed_function(*grad_output, fixed_bits=self.fixed_bits['backward'],
                                max_value=self.max_values['backward'])
        return grad_output


def linear_fixed(*input, fixed_bits, max_values, running_mode=True):
    max_values['forward'] = max_values['forward'] if 'forward' in max_values else 1
    max_values['backward'] = max_values['backward'] if 'backward' in max_values else 1
    return LinearFixedFunction(fixed_bits=fixed_bits, max_values=max_values, running_mode=running_mode)(*input)


class FixedModule(nn.Module):
    def __init__(self, module, fixed_bits={'weight':8, 'input':8, 'output':8}):
        super(FixedModule, self).__init__()
        self.module = module
        self.fixed_bits = fixed_bits

    def forward(self, input):
        input, = linear_fixed(input, 
                              fixed_bits={'forward': self.fixed_bits['input'],
                                          'backward': self.fixed_bits['input'] * 2},
                              max_values={'forward': 1, 'backward': 10})
        #float_weight = self.module.weight.clone()
        linear_fixed(self.module.weight,
                     fixed_bits={'forward': self.fixed_bits['weight'],
                                 'backward': self.fixed_bits['weight'] * 2},
                     max_values={'forward': 0.5, 'backward': 1}, running_mode=False)
        if self.module.bias is not None:
        #    float_bias = self.module.bias.clone()
            linear_fixed(self.module.bias,
                         fixed_bits={'forward': self.fixed_bits['weight'],
                                     'backward': self.fixed_bits['weight'] * 2},
                         max_values={'forward': 0.5, 'backward': 1}, running_mode=False)
        output = self.module(input)
        #self.module.weight.data = float_weight.clone()
        #if self.module.bias is not None:
        #    self.module.bias.data = float_bias.clone()
        output, = linear_fixed(output,
                               fixed_bits={'forward': self.fixed_bits['output'],
                                           'backward': self.fixed_bits['output'] * 2},
                               max_values={'forward': 10, 'backward': 1})
        return output

