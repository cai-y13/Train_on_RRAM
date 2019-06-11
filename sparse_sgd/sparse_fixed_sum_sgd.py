import torch
from torch.optim.optimizer import required, Optimizer
import math


__all__ = ['SparseFixedSumSGD']


OLD_PYTORCH = '1.12' in torch.__version__


class SparseFixedSumSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,
                 balanced_row_num={'conv':1, 'fc':1},
                 balanced_freq={'conv':1024, 'fc':128},
                 rram_size=512, small_rram_size=128,
                 is_small_by_pos=False,
                 fixed_bits={'input': 6, 'weight': 8}):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SparseFixedSumSGD, self).__init__(params, defaults)

        self.rram_size = rram_size
        self.small_rram_size = small_rram_size
        self.is_small_by_pos = is_small_by_pos
        self.fixed_step = 1 / 2 ** (fixed_bits['weight'])
        self.balanced_iter = 0
        self.balanced_row_num = balanced_row_num
        self.balanced_freq = balanced_freq
        self.row_map_m_to_r_list, self.row_map_r_to_m_list = [], []
        self.balanced_row_num_list, self.row_frequency_list, self.frequency_list = [], [], []
        print('='*89)
        for group in self.param_groups:
            # get weight size (conv/fc + bias)
            dim, n = 0, 0
            for p in group['params']:
                dim += p.data.view(p.data.size()[0], -1).size()[1]
                n = max(p.data.size()[0], n)
            # print information
            row_num = max(dim, self.rram_size)
            row_num = self.rram_size * int(math.ceil(row_num / self.rram_size))
            print('{0} : matrix = dim x n = {1} x {2} / rram = row x n = {3} x {2}'.format(
                group['config'], dim, n, row_num))
            # initial frequency_list, row_map
            self.frequency_list.append(p.data.new().long().resize_(row_num, n).zero_())
            self.row_frequency_list.append(p.data.new().long().resize_(row_num).zero_())
            self.row_map_m_to_r_list.append({i:i for i in range(row_num)})
            self.row_map_r_to_m_list.append({i:i for i in range(row_num)})
            # skip bn
            if group['config'] == 'bn':
                self.balanced_row_num_list.append(0)
                continue
            # get balanced_row_num per group
            balanced_row_num = self.balanced_row_num[group['config']]
            if balanced_row_num < 0:
                balanced_row_num = int(2 ** int(math.log2(row_num * (-balanced_row_num) // 100) + 1))
            while balanced_row_num >= row_num / 2:
                balanced_row_num = balanced_row_num // 2
            self.balanced_row_num_list.append(balanced_row_num)
        print('='*89)
        print('balanced row number:\n{0}'.format(self.balanced_row_num_list))
        print('='*89)

    def __setstate__(self, state):
        super(SparseFixedSumSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.balanced_iter += 1
        is_balanced = False

        # one group for one rram
        for group_id, group in enumerate(self.param_groups):
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            is_bn_group = group['config'] == 'bn'

            d_group = []
            dim_group = []
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = p.data.new().resize_as_(p.data).zero_()
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                if is_bn_group:
                    p.data.add_(-group['lr'], d_p)
                    continue

                if 'accumulation_buffer' not in param_state:
                    buf = param_state['accumulation_buffer'] = p.data.new().resize_as_(p.data).zero_()
                else:
                    buf = param_state['accumulation_buffer']
                buf.add_(-group['lr'], d_p)
                buf = buf.view(buf.size()[0], -1)
                d_group.append(buf)
                dim_group.append(buf.size()[1])

            if is_bn_group:
                continue

            # update weight
            d_p = torch.cat(d_group, 1)
            dim = d_p.size()[1]
            if dim <= self.small_rram_size and self.is_small_by_pos:
                importance, m_col_ids = d_p.abs().max(0)  # pytorch 2.0: size [dim] ; 1.12.0: size [1 x dim]
                if OLD_PYTORCH:
                    _, m_row_id = importance.squeeze().max(0)
                    m_row_id = m_row_id[0]
                    m_col_id = m_col_ids[0][m_row_id]
                else:
                    _, m_row_id = importance.max(0)
                    m_row_id = m_row_id[0]
                    m_col_id = m_col_ids[m_row_id]
            else:
                # large rram write in row
                if OLD_PYTORCH:
                    _, m_row_id = d_p.abs().sum(0).squeeze().max(0)
                else:
                    _, m_row_id = d_p.abs().sum(0).max(0)
                m_row_id = m_row_id[0]

            dim_p_total = 0
            for p, dim_p in zip(group['params'], dim_group):
                if m_row_id < (dim_p_total + dim_p):
                    m_row_id_p = m_row_id - dim_p_total
                    if dim <= self.small_rram_size and self.is_small_by_pos:
                        # small rram write in position
                        delta = d_p[m_col_id, m_row_id]
                        if abs(d_p) > self.fixed_bits:
                            p.data.view(p.data.size()[0], -1)[m_col_id, m_row_id_p] += delta
                            self.state[p]['accumulation_buffer'].view(p.data.size()[0], -1)[m_col_id, m_row_id_p] = 0
                            self.frequency_list[group_id][self.row_map_m_to_r_list[group_id][m_row_id], m_col_id] += 1
                            self.row_frequency_list[group_id][self.row_map_m_to_r_list[group_id][m_row_id]] += 1
                    else:
                        # large rram write in row
                        d_p = d_p.narrow(1, m_row_id, 1)  # size of [n * 1]
                        index = d_p.abs().gt(self.fixed_bits).squeeze().nonzero().squeeze()
                        if index.numel() > 0:
                            delta = d_p.index_select(0, index)
                            p.data.view(p.data.size()[0], -1).narrow(1, m_row_id_p, 1).index_add_(0, index, delta)
                            self.state[p]['accumulation_buffer'].view(p.data.size()[0], -1) \
                                                    .narrow(1, m_row_id_p, 1).index_fill_(0, index, 0)
                            self.frequency_list[group_id][self.row_map_m_to_r_list[group_id][m_row_id], :] += 1
                            self.row_frequency_list[group_id][self.row_map_m_to_r_list[group_id][m_row_id]] += 1
                    break
                dim_p_total += dim_p

            # balanced swap
            if self.balanced_iter % self.balanced_freq[group['config']] == 0:
                is_balanced = True
                _, lru = self.row_frequency_list[group_id].sort(0)
                # lru = sorted(range(dim), key=lambda i: self.frequency_list[group_id][i])
                balanced_row_num = self.balanced_row_num_list[group_id]
                mru_r_row_ids = lru[-balanced_row_num:].tolist()
                lru_r_row_ids = lru[:balanced_row_num].tolist()
                for lru_r_row_id, mru_r_row_id in zip(lru_r_row_ids, reversed(mru_r_row_ids)):
                    mru_m_row_id = self.row_map_r_to_m_list[group_id][mru_r_row_id]
                    lru_m_row_id = self.row_map_r_to_m_list[group_id][lru_r_row_id]
                    self.row_map_m_to_r_list[group_id][mru_m_row_id] = lru_r_row_id
                    self.row_map_m_to_r_list[group_id][lru_m_row_id] = mru_r_row_id
                    self.row_map_r_to_m_list[group_id][mru_r_row_id] = lru_m_row_id
                    self.row_map_r_to_m_list[group_id][lru_r_row_id] = mru_m_row_id
                    self.frequency_list[group_id][lru_r_row_id, :].add_(1)
                    self.frequency_list[group_id][mru_r_row_id, :].add_(1)
                    self.row_frequency_list[group_id][lru_r_row_id] += 1
                    self.row_frequency_list[group_id][mru_r_row_id] += 1
        if is_balanced:
            print('='*89)
            print('Swap Rows')
            print('='*89)

        return loss
