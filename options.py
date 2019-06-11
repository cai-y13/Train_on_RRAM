import datetime
import os
import argparse
import math
from collections import OrderedDict
from utils import print_section


def add_optimization_options(parser):
    parser.add_argument('--lr', '--learning-rate', dest='lr', type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--mt', '--momentum', dest='momentum',
                        type=float, metavar='M', help='momentum')
    parser.add_argument('--nesterov', action='store_true',
                        help='nesterov momentum sgd')
    parser.add_argument('--wd', '--weight-decay', dest='weight_decay',
                        type=float, metavar='WD', help='weight decay')
    parser.add_argument('--lr-decay', dest='lr_decay', type=float, metavar='LD',
                        help='every lr_decay_step epoch, learning rate decays by lr_decay'
                             ' (negative means division | positive means multiplication)')
    parser.add_argument('--lr-decay-step', dest='lr_decay_step', type=str, metavar='N1,N2...',
                        help='after N1,N2... epoch, learning rate decays by lr_decay')


def add_train_options(parser):
    parser.add_argument('data', metavar='DIR', nargs='?', help='path to dataset')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('-n', '--nGPU', dest='nGPU', default=4, type=int,
                        metavar='N', help='number of GPUs to use')
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int,
                        metavar='BS', help='mini-batch size')
    parser.add_argument('--epochs', type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pf', '--print-freq', dest='print_freq', default=10,
                        type=int, metavar='T', help='print frequency (default: 10)')
    parser.add_argument('--visdom', dest='visdom', default=False, action='store_true',
                        help='Turn on visdom graphing')

def add_sparse_train_options(parser):
    parser.add_argument('--method', dest='method', default='dft',
                        choices=['dft', 'lru', 'max', 'sum', 'flr', 'fmx', 'fsm'], metavar='SGD',
                        help='sgd type')
    parser.add_argument('--cbr', '--conv-balanced-row-num', dest='conv_balanced_row_num',
                        default=1, metavar='N', type=int,
                        help='conv_balanced_row_num for balanced_sgd (default: 1)')
    parser.add_argument('--fbr', '--fc-balanced-row-num', dest='fc_balanced_row_num',
                        default=1, metavar='N', type=int,
                        help='fc_balanced_row_num for balanced_sgd (default: 1)')
    parser.add_argument('--cbf', '--conv-balanced-freq', dest='conv_balanced_freq',
                        default=1024, metavar='N', type=int,
                        help='conv_balanced_freq for balanced_sgd (default: 1024)')
    parser.add_argument('--fbf', '--fc-balanced-freq', dest='fc_balanced_freq',
                        default=1024, metavar='N', type=int,
                        help='fc_balanced_freq for balanced_sgd (default: 1024)')
    parser.add_argument('-s', '--rram-size', dest='rram_size',
                        default=128, metavar='N', type=int,
                        help='minimum rram size (default: 128)')
    parser.add_argument('--ss', '--small-rram-size', dest='small_rram_size',
                        default=128, metavar='N', type=int,
                        help='threshold for small rram size (default: 128)')
    parser.add_argument('--sp', '--small-by-pos', dest='is_small_by_pos',
                        default=False, action='store_true',
                        help='Turn on update small matrix by position')
    parser.add_argument('--ib', '--input-bits', dest='input_bits',
                        default=8, metavar='N', type=int,
                        help='input fixed bits (default: 8)')
    parser.add_argument('--ob', '--output-bits', dest='output_bits',
                        default=8, metavar='N', type=int,
                        help='output fixed bits (default: 8)')
    parser.add_argument('--wb', '--weight-bits', dest='weight_bits',
                        default=8, metavar='N', type=int,
                        help='weight fixed bits (default: 8)')
    parser.add_argument('-l', dest='is_log_frequency', default=False,
                        action='store_true', help='whether log update frequency')
    parser.add_argument('--lf', '--log-freq', dest='log_freq', default=40,
                        type=int, metavar='T', help='log frequency (default: 40)')


def create_config(**options):
    config = OrderedDict()
    config['arch'] = options['arch']
    config['lr'] = options['lr']
    config['momentum'] = options['momentum'] if 'momentum' in options else 0.0
    config['nesterov'] = options['nesterov'] if 'nesterov' in options else False
    config['weight_decay'] = options['weight_decay'] if 'weight_decay' in options else 0.0
    config['lr_decay'] = options['lr_decay'] if 'lr_decay' in options else 1.0
    if config['lr_decay'] < 0:
        config['lr_decay'] = -1.0 / config['lr_decay']
    config['lr_decay_step'] = options['lr_decay_step'] if 'lr_decay_step' in options else -1.0
    if isinstance(config['lr_decay_step'], str):
        lr_decay_step = config['lr_decay_step'].split(',')
        if len(lr_decay_step) == 1:
            lr_decay_step = int(lr_decay_step[0])
        else:
            if not lr_decay_step[-1]:
                lr_decay_step = lr_decay_step[:-1]
            lr_decay_step = [int(x) for x in lr_decay_step]
        config['lr_decay_step'] = lr_decay_step

    config['method'] = options['method'] if 'method' in options else 'dft'
    if 'fixed' not in config['arch']:
        if config['method'] == 'flr':
            config['method'] = 'lru'
        elif config['method'] == 'fmx':
            config['method'] = 'max'
        elif config['method'] == 'fsm':
            config['method'] = 'sum'
    config['conv_balanced_row_num'] = options['conv_balanced_row_num'] \
                                        if 'conv_balanced_row_num' in options else 1
    config['fc_balanced_row_num'] = options['fc_balanced_row_num'] \
                                        if 'fc_balanced_row_num' in options else 1
    config['conv_balanced_freq'] = options['conv_balanced_freq'] \
                                        if 'conv_balanced_freq' in options else 1024
    config['fc_balanced_freq'] = options['fc_balanced_freq'] \
                                        if 'fc_balanced_freq' in options else 1024
    config['rram_size'] = options['rram_size'] if 'rram_size' in options else 512
    config['small_rram_size'] = options['small_rram_size'] \
                                        if 'small_rram_size' in options else 128
    config['is_small_by_pos'] = options['is_small_by_pos'] \
                                        if 'is_small_by_pos' in options else False
    config['input_bits'] = options['input_bits'] if 'input_bits' in options else 8
    config['output_bits'] = options['output_bits'] if 'output_bits' in options else 8
    config['weight_bits'] = options['weight_bits'] if 'weight_bits' in options else 8

    config_str = 'Training Config:\n'
    for key, val in config.items():
        config_str += '{}: {}\n'.format(key, val)

    print_section(config_str, up=89, down=False)

    alg_id = config['arch'] + '_' + config['method'] + \
             '_c_' + str(config['conv_balanced_row_num']) + \
             '_t_' + str(config['fc_balanced_row_num']) + \
             '_b_' + str(config['conv_balanced_freq']) + \
             '_f_' + str(config['fc_balanced_freq']) + \
             '_r_' + str(config['rram_size']) + \
             '_p_' + str(1 if config['is_small_by_pos'] else 0) + \
             '_h_' + str(config['small_rram_size']) + '_' + \
             datetime.datetime.now().strftime('%m%d_%H%M')

    print_section('saving to %s' % alg_id, up=False, down=89)

    config['checkpoint_dir'] = os.path.join('checkpoints', alg_id)
    config['log_dir'] = os.path.join('logs', alg_id)
    config['experiment_id'] = alg_id
    config['config_str'] = config_str

    return config


class Options:
    def __init__(self, description):
        self.parser = argparse.ArgumentParser(description=description)
        add_optimization_options(self.parser)
        add_train_options(self.parser)
        add_sparse_train_options(self.parser)
        self.config_str = None

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def set_defaults(self, **kwargs):
        self.parser.set_defaults(**kwargs)

    def parse_args(self, *args):
        args = self.parser.parse_args(*args)
        vars(args)['correction_coeff'] = args.momentum
        for k, v in vars(args).items():
            self.__setattr__(k, v)
        return self

    def set_config(self, key, value):
        self.__setattr__(key, value)
        return self

    def get_config(self):
        if self.config_str is None:
            config = create_config(**vars(self))
            for k, v in config.items():
                self.__setattr__(k, v)
        return vars(self)
