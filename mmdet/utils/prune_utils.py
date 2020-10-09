import torch
import torch.nn as nn
# from mmdet.models.backbones import channel_selection
#
# class BNOptimizer():
#
#     @staticmethod
#     def updateBN(model, s):
#         model_backbone = model.backbone.modules()
#         for layer_id, m in enumerate(model_backbone):   # only prune backbone
#             if isinstance(m, nn.BatchNorm2d):
#                 if not isinstance(model_backbone[layer_id+1], channel_selection):
#                     m.weight.grad.data.add_(s * torch.sign(m.weight.data))  # L1