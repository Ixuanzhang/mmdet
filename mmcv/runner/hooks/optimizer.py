# Copyright (c) Open-MMLab. All rights reserved.
from torch.nn.utils import clip_grad

from .hook import Hook
import torch
import torch.nn as nn
from mmcv.cnn import channel_selection

class BNOptimizer():

    @staticmethod
    def updateBN(model, s):
        model_backbone = list(model.backbone.modules())
        for layer_id, m in enumerate(model_backbone):   # only prune backbone
            if isinstance(m, nn.BatchNorm2d):
                if not isinstance(model_backbone[layer_id+1], channel_selection):
                    m.weight.grad.data.add_(s * torch.sign(m.weight.data))  # L1

class OptimizerHook(Hook):

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        clip_grad.clip_grad_norm_(
            filter(lambda p: p.requires_grad, params), **self.grad_clip)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            self.clip_grads(runner.model.parameters())
        
        # 稀疏训练
        if runner.sr:
            BNOptimizer.updateBN(runner.model.module, runner.s)
        
        runner.optimizer.step()
