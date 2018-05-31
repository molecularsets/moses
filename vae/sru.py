""" SRU Implementation """
# flake8: noqa

import subprocess
import platform
import os
import re
import argparse
import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from collections import namedtuple


# For command-line option parsing
class CheckSRU(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super(CheckSRU, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values == 'SRU':
            check_sru_requirement(abort=True)
        # Check pass, set the args.
        setattr(namespace, self.dest, values)


# This SRU version implements its own cuda-level optimization,
# so it requires that:
# 1. `cupy` and `pynvrtc` python package installed.
# 2. pytorch is built with cuda support.
# 3. library path set: export LD_LIBRARY_PATH=<cuda lib path>.
def check_sru_requirement(abort=False):
    """
    Return True if check pass; if check fails and abort is True,
    raise an Exception, othereise return False.
    """

    # Check 1.
    try:
        if platform.system() == 'Windows':
            subprocess.check_output('pip freeze | findstr cupy', shell=True)
            subprocess.check_output('pip freeze | findstr pynvrtc',
                                    shell=True)
        else:  # Unix-like systems
            subprocess.check_output('pip freeze | grep -w cupy', shell=True)
            subprocess.check_output('pip freeze | grep -w pynvrtc',
                                    shell=True)
    except subprocess.CalledProcessError:
        if not abort:
            return False
        raise AssertionError("Using SRU requires 'cupy' and 'pynvrtc' "
                             "python packages installed.")

    # Check 2.
    if torch.cuda.is_available() is False:
        if not abort:
            return False
        raise AssertionError("Using SRU requires pytorch built with cuda.")

    # Check 3.
    pattern = re.compile(".*cuda/lib.*")
    ld_path = os.getenv('LD_LIBRARY_PATH', "")
    if re.match(pattern, ld_path) is None:
        if not abort:
            return False
        raise AssertionError("Using SRU requires setting cuda lib path, e.g. "
                             "export LD_LIBRARY_PATH=/usr/local/cuda/lib64.")

    return True


SRU_CODE = open('sru.cu').read()
SRU_FWD_FUNC, SRU_BWD_FUNC = None, None
SRU_BiFWD_FUNC, SRU_BiBWD_FUNC = None, None
SRU_STREAM = None


def load_sru_mod():
    global SRU_FWD_FUNC, SRU_BWD_FUNC, SRU_BiFWD_FUNC, SRU_BiBWD_FUNC
    global SRU_STREAM
    if check_sru_requirement():
        from cupy.cuda import function, Device
        # Device(device.index).use()
        from pynvrtc.compiler import Program

        # This sets up device to use.
        # tmp_ = torch.rand(1, 1).to(device)

        sru_prog = Program(SRU_CODE.encode('utf-8'),
                           'sru_prog.cu'.encode('utf-8'))
        sru_ptx = sru_prog.compile()
        sru_mod = function.Module()
        sru_mod.load(bytes(sru_ptx.encode()))

        SRU_FWD_FUNC = sru_mod.get_function('sru_fwd')
        SRU_BWD_FUNC = sru_mod.get_function('sru_bwd')
        SRU_BiFWD_FUNC = sru_mod.get_function('sru_bi_fwd')
        SRU_BiBWD_FUNC = sru_mod.get_function('sru_bi_bwd')

        stream = namedtuple('Stream', ['ptr'])
        SRU_STREAM = stream(ptr=torch.cuda.current_stream().cuda_stream)


class SRU_Compute(Function):

    def __init__(self, activation_type, d_out, bidirectional=False):
        SRU_Compute.maybe_load_sru_mod()
        super(SRU_Compute, self).__init__()
        self.activation_type = activation_type
        self.d_out = d_out
        self.bidirectional = bidirectional

    @staticmethod
    def maybe_load_sru_mod():
        global SRU_FWD_FUNC

        if SRU_FWD_FUNC is None:
            load_sru_mod()

    def forward(self, u, x, bias, init=None, mask_h=None):
        bidir = 2 if self.bidirectional else 1
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = self.d_out
        k = u.size(-1) // d
        k_ = k // 2 if self.bidirectional else k
        ncols = batch * d * bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols - 1) // thread_per_block + 1

        init_ = x.new(ncols).zero_() if init is None else init
        size = (length, batch, d * bidir) if x.dim() == 3 else (
        batch, d * bidir)
        c = x.new(*size)
        h = x.new(*size)

        FUNC = SRU_FWD_FUNC if not self.bidirectional else SRU_BiFWD_FUNC
        FUNC(args=[
            u.contiguous().data_ptr(),
            x.contiguous().data_ptr() if k_ == 3 else 0,
            bias.data_ptr(),
            init_.contiguous().data_ptr(),
            mask_h.data_ptr() if mask_h is not None else 0,
            length,
            batch,
            d,
            k_,
            h.data_ptr(),
            c.data_ptr(),
            self.activation_type],
            block=(thread_per_block, 1, 1), grid=(num_block, 1, 1),
            stream=SRU_STREAM
        )

        self.save_for_backward(u, x, bias, init, mask_h)
        self.intermediate = c
        if x.dim() == 2:
            last_hidden = c
        elif self.bidirectional:
            # -> directions x batch x dim
            last_hidden = torch.stack((c[-1, :, :d], c[0, :, d:]))
        else:
            last_hidden = c[-1]
        return h, last_hidden

    def backward(self, grad_h, grad_last):
        if self.bidirectional:
            grad_last = torch.cat((grad_last[0], grad_last[1]), 1)
        bidir = 2 if self.bidirectional else 1
        u, x, bias, init, mask_h = self.saved_tensors
        c = self.intermediate
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = self.d_out
        k = u.size(-1) // d
        k_ = k // 2 if self.bidirectional else k
        ncols = batch * d * bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols - 1) // thread_per_block + 1

        init_ = x.new(ncols).zero_() if init is None else init
        grad_u = u.new(*u.size())
        grad_bias = x.new(2, batch, d * bidir)
        grad_init = x.new(batch, d * bidir)

        # For DEBUG
        # size = (length, batch, x.size(-1)) \
        #         if x.dim() == 3 else (batch, x.size(-1))
        # grad_x = x.new(*x.size()) if k_ == 3 else x.new(*size).zero_()

        # Normal use
        grad_x = x.new(*x.size()) if k_ == 3 else None

        FUNC = SRU_BWD_FUNC if not self.bidirectional else SRU_BiBWD_FUNC
        FUNC(args=[
            u.contiguous().data_ptr(),
            x.contiguous().data_ptr() if k_ == 3 else 0,
            bias.data_ptr(),
            init_.contiguous().data_ptr(),
            mask_h.data_ptr() if mask_h is not None else 0,
            c.data_ptr(),
            grad_h.contiguous().data_ptr(),
            grad_last.contiguous().data_ptr(),
            length,
            batch,
            d,
            k_,
            grad_u.data_ptr(),
            grad_x.data_ptr() if k_ == 3 else 0,
            grad_bias.data_ptr(),
            grad_init.data_ptr(),
            self.activation_type],
            block=(thread_per_block, 1, 1), grid=(num_block, 1, 1),
            stream=SRU_STREAM
        )
        return grad_u, grad_x, grad_bias.sum(1).view(-1), grad_init, None


class SRUCell(nn.Module):
    def __init__(self, n_in, n_out, dropout=0, rnn_dropout=0,
                 bidirectional=False, use_tanh=1, use_relu=0):
        super(SRUCell, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.rnn_dropout = rnn_dropout
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.activation_type = 2 if use_relu else (1 if use_tanh else 0)

        out_size = n_out * 2 if bidirectional else n_out
        k = 4 if n_in != out_size else 3
        self.size_per_dir = n_out * k
        self.weight = nn.Parameter(torch.Tensor(
            n_in,
            self.size_per_dir * 2 if bidirectional else self.size_per_dir
        ))
        self.bias = nn.Parameter(torch.Tensor(
            n_out * 4 if bidirectional else n_out * 2
        ))
        self.init_weight()

    def init_weight(self):
        val_range = (3.0 / self.n_in) ** 0.5
        self.weight.data.uniform_(-val_range, val_range)
        self.bias.data.zero_()

    def set_bias(self, bias_val=0):
        n_out = self.n_out
        if self.bidirectional:
            self.bias.data[n_out * 2:].zero_().add_(bias_val)
        else:
            self.bias.data[n_out:].zero_().add_(bias_val)

    def forward(self, input, c0=None):
        assert input.dim() == 2 or input.dim() == 3
        n_in, n_out = self.n_in, self.n_out
        batch = input.size(-2)
        if c0 is None:
            c0 = Variable(input.data.new(
                batch, n_out if not self.bidirectional else n_out * 2
            ).zero_())

        if self.training and (self.rnn_dropout > 0):
            mask = self.get_dropout_mask_((batch, n_in), self.rnn_dropout)
            x = input * mask.expand_as(input)
        else:
            x = input

        x_2d = x if x.dim() == 2 else x.contiguous().view(-1, n_in)
        u = x_2d.mm(self.weight)

        if self.training and (self.dropout > 0):
            bidir = 2 if self.bidirectional else 1
            mask_h = self.get_dropout_mask_((batch, n_out * bidir),
                                            self.dropout)
            h, c = SRU_Compute(self.activation_type, n_out,
                               self.bidirectional)(
                u, input, self.bias, c0, mask_h
            )
        else:
            h, c = SRU_Compute(self.activation_type, n_out,
                               self.bidirectional)(
                u, input, self.bias, c0
            )

        return h, c

    def get_dropout_mask_(self, size, p):
        w = self.weight.data
        return Variable(w.new(*size).bernoulli_(1 - p).div_(1 - p))


class SRU(nn.Module):
    """
    Implementation of "Training RNNs as Fast as CNNs"
    :cite:`DBLP:journals/corr/abs-1709-02755`

    TODO: turn to pytorch's implementation when it is available.

    This implementation is adpoted from the author of the paper:
    https://github.com/taolei87/sru/blob/master/cuda_functional.py.

    Args:
      input_size (int): input to model
      hidden_size (int): hidden dimension
      num_layers (int): number of layers
      dropout (float): dropout to use (stacked)
      rnn_dropout (float): dropout to use (recurrent)
      bidirectional (bool): bidirectional
      use_tanh (bool): activation
      use_relu (bool): activation

    """

    def __init__(self, input_size, hidden_size,
                 num_layers=2, dropout=0, rnn_dropout=0,
                 bidirectional=False, use_tanh=1, use_relu=0):
        # An entry check here, will catch on train side and translate side
        # if requirements are not satisfied.
        check_sru_requirement(abort=True)
        super(SRU, self).__init__()
        self.n_in = input_size
        self.n_out = hidden_size
        self.depth = num_layers
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.rnn_lst = nn.ModuleList()
        self.bidirectional = bidirectional
        self.out_size = hidden_size * 2 if bidirectional else hidden_size

        for i in range(num_layers):
            sru_cell = SRUCell(
                n_in=self.n_in if i == 0 else self.out_size,
                n_out=self.n_out,
                dropout=dropout if i + 1 != num_layers else 0,
                rnn_dropout=rnn_dropout,
                bidirectional=bidirectional,
                use_tanh=use_tanh,
                use_relu=use_relu,
            )
            self.rnn_lst.append(sru_cell)

    def set_bias(self, bias_val=0):
        for l in self.rnn_lst:
            l.set_bias(bias_val)

    def forward(self, input, c0=None, return_hidden=True):
        assert input.dim() == 3  # (len, batch, n_in)
        dir_ = 2 if self.bidirectional else 1
        if c0 is None:
            zeros = Variable(input.data.new(
                input.size(1), self.n_out * dir_
            ).zero_())
            c0 = [zeros for i in range(self.depth)]
        else:
            if isinstance(c0, tuple):
                # RNNDecoderState wraps hidden as a tuple.
                c0 = c0[0]
            assert c0.dim() == 3  # (depth, batch, dir_*n_out)
            c0 = [h.squeeze(0) for h in c0.chunk(self.depth, 0)]

        prevx = input
        lstc = []
        for i, rnn in enumerate(self.rnn_lst):
            h, c = rnn(prevx, c0[i])
            prevx = h
            lstc.append(c)

        if self.bidirectional:
            # fh -> (layers*directions) x batch x dim
            fh = torch.cat(lstc)
        else:
            fh = torch.stack(lstc)

        if return_hidden:
            return prevx, fh
        else:
            return prevx
