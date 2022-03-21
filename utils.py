#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        : 2022/3/21 12:59
# @Author      : sgallon
# @Email       : shcmsgallon@outlook.com
# @File        : utils.py
# @Description :

import torch
from torch import nn
from functools import partial
from fastai.text.all import *


class MyConfig(dict):
    def __getattr__(self, name): return self[name]

    def __setattr__(self, name, value): self[name] = value


@delegates()
class MyMSELossFlat(BaseLoss):
    def __init__(self, *args, axis=-1, floatify=True, low=None, high=None, **kwargs):
        super().__init__(nn.MSELoss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)
        self.low, self.high = low, high

    def decodes(self, x):
        if self.low is not None:
            x = torch.max(x, x.new_full(x.shape, self.low))
        if self.high is not None:
            x = torch.min(x, x.new_full(x.shape, self.high))
        return x


def adam_no_correction_step(p, lr, mom, step, sqr_mom, grad_avg, sqr_avg, eps, **kwargs):
    p.data.addcdiv_(grad_avg, (sqr_avg).sqrt() + eps, value=-lr)
    return p


def Adam_no_bias_correction(params, lr, mom=0.9, sqr_mom=0.99, eps=1e-5, wd=0.01, decouple_wd=True):
    "An `Optimizer` for Adam with `lr`, `mom`, `sqr_mom`, `eps` and `params`"
    cbs = [weight_decay] if decouple_wd else [l2_reg]
    cbs += [partial(average_grad, dampening=True), average_sqr_grad, step_stat, adam_no_correction_step]
    return Optimizer(params, cbs, lr=lr, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd)


class Ensemble(nn.Module):
    def __init__(self, models, device='cuda:0', merge_out_fc=None):
        super().__init__()
        self.models = nn.ModuleList(m.cpu() for m in models)
        self.device = device
        self.merge_out_fc = merge_out_fc

    def to(self, device):
        self.device = device
        return self

    def getitem(self, i):
        return self.models[i]

    def forward(self, *args, **kwargs):
        outs = []
        for m in self.models:
            m.to(self.device)
            out = m(*args, **kwargs)
            m.cpu()
            outs.append(out)
        if self.merge_out_fc:
            outs = self.merge_out_fc(outs)
        else:
            outs = torch.stack(outs)
            outs = outs.mean(dim=0)
        return outs


def load_model_(learn, files, device=None, **kwargs):
    "if multiple file passed, then load and create an ensemble. Load normally otherwise"
    merge_out_fc = kwargs.pop('merge_out_fc', None)
    if not isinstance(files, list):
        learn.load(files, device=device, **kwargs)
        return
    if device is None: device = learn.dls.device
    model = learn.model.cpu()
    models = [model, *(deepcopy(model) for _ in range(len(files) - 1))]
    for f, m in zip(files, models):
        file = join_path_file(f, learn.path / learn.model_dir, ext='.pth')
        load_model(file, m, learn.opt, device='cpu', **kwargs)
    learn.model = Ensemble(models, device, merge_out_fc)
    return learn


class GradientClipping(Callback):
    def __init__(self, clip: float = 0.1):
        self.clip = clip
        assert self.clip

    def after_backward(self):
        if hasattr(self, 'scaler'): self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
