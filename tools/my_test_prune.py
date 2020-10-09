#-*-coding:utf-8-*-
import argparse
import os

import mmcv
import torch
import torch.nn as nn
import numpy as np
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
# from mmdet.models.backbones import channel_selection
from mmdet.models.backbones import PResNet
from mmcv.cnn import channel_selection

import copy


class MultipleKVAction(argparse.Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary.
    """

    def _is_int(self, val):
        try:
            _ = int(val)
            return True
        except Exception:
            return False

    def _is_float(self, val):
        try:
            _ = float(val)
            return True
        except Exception:
            return False

    def _is_bool(self, val):
        return val.lower() in ['true', 'false']

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for val in values:
            parts = val.split('=')
            key = parts[0].strip()
            if len(parts) > 2:
                val = '='.join(parts[1:])
            else:
                val = parts[1].strip()
            # try parsing val to bool/int/float first
            if self._is_bool(val):
                import json
                val = json.loads(val.lower())
            elif self._is_int(val):
                val = int(val)
            elif self._is_float(val):
                val = float(val)
            options[key] = val
        setattr(namespace, self.dest, options)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format_only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=MultipleKVAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    # add prune parser
    parser.add_argument('--percent', type=float, default=0.25,
                        help='scale sparse rate (default: 0.5)')
    parser.add_argument('--save_path', default='/home/stella/zx/mmdetection-master/result_out/prune', type=str, metavar='PATH',
                        help='path to save pruned model (default: none)')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results) with the argument "--out", "--eval", "--format_only" '
         'or "--show"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    # 计算需要剪枝的变量个数total
    model.cuda()
    total = 0
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    # 确定剪枝的全局阈值
    bn = torch.zeros(total)
    index = 0
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size

    # 按照权值大小排序
    y, i = torch.sort(bn)
    thre_index = int(total * args.percent)

    # 确定要剪枝的阈值
    thre = y[thre_index].cuda()

    # ********************************预剪枝*********************************#
    pruned = 0
    cfg_ori = []
    cfg = []
    cfg_mask = []
    model_backbone = list(model.backbone.modules())
    for layer_id, m in enumerate(model_backbone):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            if isinstance(model_backbone[layer_id+1], channel_selection):
                mask = torch.ones(weight_copy.shape[0]).cuda()
            else:
                # 要保留的通道标记Mask图
                mask = weight_copy.gt(thre).float().cuda()
                # 要保留的通道标记Mask图
                pruned = pruned + mask.shape[0] - torch.sum(mask)
            # m.weight.data.mul_(mask)
            # m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_ori.append(mask.shape[0])
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(layer_id, mask.shape[0], int(torch.sum(mask))))

    pruned_ratio = pruned / total

    print("剪枝比例：")
    print(pruned_ratio)

    print('Pre-processing Successful!')

    print('cfg:')
    print(cfg)

    # ******************************* 正式剪枝 ********************************#
    # 每个阶的最一层不剪枝
    newmodel = copy.deepcopy(model)
    newmodel.backbone = PResNet(
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        cfg = cfg)
    newmodel.cuda()

    # print(newmodel.backbone)

    num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    savepath = os.path.join(args.save_path, "prune.txt")
    # with open(savepath, "w") as fp:
    #     fp.write("Configuration: \n" + str(cfg) + "\n")
    #     fp.write("Number of parameters: \n" + str(num_parameters) + "\n")
    #     fp.write("Test accuracy: \n" + str(acc))

    old_modules = list(model.backbone.modules())
    new_modules = list(newmodel.backbone.modules())
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    conv_count = 0
    # downsample_conv_list = [17, 48, 88, 299]
    downsample_conv_list = [17, 49, 90, 302]

    for layer_id, m0 in enumerate(old_modules):
        # m0 = old_modules[layer_id]
        # print('m0:')
        # print(m0)
        m1 = new_modules[layer_id]
        # print('m1:')
        # print(m1)
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))

            if isinstance(old_modules[layer_id + 1], channel_selection):
                # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                # We need to set the channel selection layer.
                m2 = new_modules[layer_id + 1]
                m2.indexes.data.zero_()
                m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
            else:
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            if conv_count == 0:
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue
            if layer_id in downsample_conv_list:
                # We need to consider the case where there are downsampling convolutions.
                # For these convolutions, we just copy the weights.
                m1.weight.data = m0.weight.data.clone()
                continue
            if isinstance(old_modules[layer_id + 1], nn.BatchNorm2d):
                # This convers the convolutions in the residual block.
                # The convolutions are either after the channel selection layer or after the batch normalization layer.
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()

                # If the current convolution is not the last convolution in the residual block, then we can change the
                # number of output channels. Currently we use `conv_count` to detect whether it is such convolution.
                # if conv_count % 3 != 0:
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
                conv_count += 1
                continue


        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))

            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()

    # torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save_path, 'pruned.pth.tar'))
    # torch.save(newmodel.state_dict(), os.path.join(args.save_path, 'pruned.pth'))
    print(newmodel)
    torch.save(newmodel, os.path.join(args.save_path, 'pruned.pth'))

    # print(newmodel)

    if not distributed:
        newmodel = MMDataParallel(newmodel, device_ids=[0])
        newmodel = single_gpu_test(newmodel, data_loader, args.show)
    else:
        newmodel = MMDistributedDataParallel(
            newmodel.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(newmodel, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print('\nwriting results to {}'.format(args.out))
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.options is None else args.options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            dataset.evaluate(outputs, args.eval, **kwargs)


if __name__ == '__main__':
    main()
