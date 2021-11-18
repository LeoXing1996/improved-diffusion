import argparse
import copy
import os
import os.path as osp

import torch
from apis.train import set_random_seed
from improved_diffusion.image_datasets import build_dataloader, build_dataset
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DDPM model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    # learn_sigma = cfg.get('learn_sigma', False)
    data_cfg_ = copy.deepcopy(cfg.data)

    if hasattr(data_cfg_, 'type') and data_cfg_.type == 'RepeatDataset':
        from improved_diffusion.image_datasets import RepeatDataset
        times = data_cfg_.times
        dataset = RepeatDataset(
            build_dataset(**data_cfg_.dataset, launcher=args.launcher),
            times=times)
    else:
        dataset = build_dataset(**data_cfg_, launcher=args.launcher)

    dataloader = build_dataloader(
        dataset,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        # cfg.gpus will be ignored if distributed
        len(cfg.gpu_ids),
        dist=distributed,
        persistent_workers=cfg.data.get('persistent_workers', False),
        seed=cfg.seed)

    print(len(dataloader.dataset) // cfg.data.samples_per_gpu)
    for idx, data in enumerate(tqdm(dataloader)):
        if idx > len(dataloader.dataset) // cfg.data.samples_per_gpu:
            break


if __name__ == '__main__':
    main()
