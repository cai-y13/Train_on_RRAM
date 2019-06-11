import torch
import torch.nn as nn
from .rram import FixedModule

def profile_conv(config, rram_size={'row':128, 'col':128}):
    param_map = dict()
    channel_per_crossbar = rram_size['row'] // (config['kernel_size'] ** 2)
    row_block = config['in_channel'] // channel_per_crossbar 
    row_left = config['in_channel'] % channel_per_crossbar
    col_block = config['out_channel'] // rram_size['col']
    col_left = config['out_channel'] % rram_size['col']
    param_map['row_block'] = row_block if (row_left == 0) else (row_block+1)
    param_map['col_block'] = col_block if (col_left == 0) else (col_block+1)
    param_map['split_row'] = [channel_per_crossbar] * row_block
    param_map['split_col'] = [rram_size['col']] * col_block
    if row_left != 0:
        param_map['split_row'].append(row_left)
    if col_left != 0:
        param_map['split_col'].append(col_left)
    print(param_map)
    return param_map   

def profile_fc(config, rram_size={'row':128, 'col':128}):
    param_map = dict()
    row_block = config['in_neuron'] // rram_size['row']
    row_left = config['in_neuron'] % rram_size['row']
    col_block = config['out_neuron'] // rram_size['col']
    col_left = config['out_neuron'] % rram_size['col']
    param_map['row_block'] = row_block if (row_left == 0) else (row_block+1)
    param_map['col_block'] = col_block if (col_left == 0) else (col_block+1)
    param_map['split_row'] = [rram_size['row']] * row_block
    param_map['split_col'] = [rram_size['col']] * col_block
    if row_left != 0:
        param_map['split_row'].append(row_left)
    if col_left != 0:
        param_map['split_col'].append(col_left)
    print(param_map)
    return param_map 
     

class SplitConv(nn.Module):
    def __init__(self, config, rram_size={'row':128, 'col':128}):
        super(SplitConv, self).__init__()
        self.profile = profile_conv(config) 
        self.rram_size = rram_size
        self.sub_convs = []
        for i in range(self.profile['row_block']):
            sub_conv = FixedModule(nn.Conv2d(self.profile['split_row'][i], config['out_channel'], \
                           kernel_size=config['kernel_size'], stride=config['stride'], \
                           padding=config['padding'], bias=config['bias']), \
                           fixed_bits=config['fixed_bits'])
            self.sub_convs.append(sub_conv)
        self.sub_convs = nn.ModuleList(self.sub_convs)
    def forward(self, x):
        accum_channel = 0
        for i in range(self.profile['row_block']):
            sub_out = self.sub_convs[i](x[:, accum_channel:accum_channel+self.profile['split_row'][i], :, :])
            accum_channel += self.profile['split_row'][i]
            if i == 0:
                out = sub_out.clone()
            else:
                out += sub_out
        return out

class SplitFC(nn.Module):
    def __init__(self, config, rram_size={'row':128, 'col':128}):
        super(SplitFC, self).__init__()
        self.profile = profile_fc(config)
        self.rram_size = rram_size
        self.sub_fcs = []
        for i in range(self.profile['row_block']):
            sub_fc = FixedModule(nn.Linear(self.profile['split_row'][i], config['out_neuron']), fixed_bits=config['fixed_bits'])
            self.sub_fcs.append(sub_fc)
        self.sub_fcs = nn.ModuleList(self.sub_fcs)
    def forward(self, x):
        accum_neuron = 0
        for i in range(self.profile['row_block']):
            sub_out = self.sub_fcs[i](x[:, accum_neuron:accum_neuron+self.profile['split_row'][i]])
            accum_neuron += self.profile['split_row'][i]
            if i == 0:
                out = sub_out.clone()
            else:
                out += sub_out
        return out
