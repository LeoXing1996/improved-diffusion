import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
from apis.train import collect_env, get_root_logger, set_random_seed
from improved_diffusion.image_datasets import build_dataloader, build_dataset
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (create_gaussian_diffusion,
                                            create_model)
from improved_diffusion.train_util import TrainLoop
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from pavi import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
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

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    # learn_sigma = cfg.get('learn_sigma', False)
    learn_sigma = cfg.model_and_diffusion.get('learn_sigma', False)
    model_cfg_ = copy.deepcopy(cfg.model_and_diffusion.model)
    model_cfg_['learn_sigma'] = learn_sigma
    diffusion_cfg_ = copy.deepcopy(cfg.model_and_diffusion.diffusion)
    diffusion_cfg_['learn_sigma'] = learn_sigma

    assert 'train_cfg' in cfg
    train_cfg_ = copy.deepcopy(cfg.train_cfg)

    model = create_model(**model_cfg_)
    diffusion = create_gaussian_diffusion(**diffusion_cfg_)
    sampler = create_named_schedule_sampler(
        train_cfg_.get('schedule_sampler', 'uniform'), diffusion)
    train_cfg_['schedule_sampler'] = sampler
    writer_cfg = cfg.writer
    train_cfg_['mm_writer'] = SummaryWriter(**writer_cfg)
    train_cfg_['save_dir'] = osp.join(cfg.work_dir, train_cfg_['save_dir'])

    data_cfg_ = copy.deepcopy(cfg.data)

    import ipdb
    ipdb.set_trace()
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

    train_cfg_['batch_size'] = cfg.data.samples_per_gpu
    train_cfg_['resume_checkpoint'] = args.resume_from
    train_looper = TrainLoop(
        model=model, diffusion=diffusion, data=dataloader, **train_cfg_)
    train_looper.run_loop()


if __name__ == '__main__':
    main()
