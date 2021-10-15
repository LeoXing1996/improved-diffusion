"""Generate a large batch of image samples from a model and save them as a
large numpy array.

This can be used to produce samples for FID evaluation.
"""

import argparse
import copy
import os
import os.path as osp

import mmcv
import numpy as np
import torch as th
import torch.distributed as dist
from apis.train import get_root_logger, set_random_seed
from improved_diffusion import dist_util
from improved_diffusion.script_util import (create_gaussian_diffusion,
                                            create_model)
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a DDPM model')
    parser.add_argument('config', help='evaluation config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--batch-size', type=int, default=100, help='batch size of dataloader')
    parser.add_argument(
        '--samples-path',
        type=str,
        default=None,
        help='path to store images. If not given, remove it after evaluation\
             finished')
    parser.add_argument(
        '--sample-model',
        type=str,
        default='ema',
        help='use which mode (ema/orig) in sampling')
    parser.add_argument(
        '--eval',
        nargs='*',
        type=str,
        default=None,
        help='select the metrics you want to access')
    parser.add_argument(
        '--online',
        action='store_true',
        help='whether to use online mode for evaluation')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=-1,
        help='whether to use online mode for evaluation')
    parser.add_argument('--work-dir', default=None)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--sample-cfg',
        nargs='+',
        action=DictAction,
        help='Other customized kwargs for sampling function')
    parser.add_argument('--clip-denoised', default=True)
    parser.add_argument('--use-ddim', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        th.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0],
                                'sample')
    os.makedirs(cfg.work_dir, exist_ok=True)

    if args.num_samples == -1:
        args.num_samples = 50000

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()

    dirname = os.path.dirname(args.checkpoint)
    ckpt = os.path.basename(args.checkpoint)

    if 'http' in args.checkpoint:
        log_path = None
    else:
        log_name = ckpt.split('.')[0] + '_eval_log' + '.txt'
        log_path = os.path.join(dirname, log_name)

    logger = get_root_logger(log_file=log_path, file_mode='a')
    logger.info('evaluation')

    # set random seeds
    if args.seed is not None:
        if rank == 0:
            mmcv.print_log(f'set random seed to {args.seed}', 'mmgen')
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the model and load checkpoint

    learn_sigma = cfg.model_and_diffusion.get('learn_sigma', False)
    model_cfg_ = copy.deepcopy(cfg.model_and_diffusion.model)
    model_cfg_['learn_sigma'] = learn_sigma
    diffusion_cfg_ = copy.deepcopy(cfg.model_and_diffusion.diffusion)
    diffusion_cfg_['learn_sigma'] = learn_sigma

    model = create_model(**model_cfg_)
    diffusion = create_gaussian_diffusion(**diffusion_cfg_)

    model.load_state_dict(
        dist_util.load_state_dict(args.checkpoint, map_location='cpu'))
    if th.cuda.is_available():
        # TODO: support DDP
        model.cuda()

    # get sample configs
    num_classes = cfg.model_and_diffusion.get('num_classes', 1000)
    image_size = cfg.data.image_size

    # logger.info(f'Sampling model: {args.sample_model}', 'mmgen')

    model.eval()

    all_images = []
    all_labels = []

    # TODO: use hard code instead config here -->
    #   should convert args.num_samples to config
    #   or use metric num_samples instead
    if rank == 0:
        pbar = mmcv.ProgressBar(args.num_samples)

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if num_classes > 0:
            classes = th.randint(
                low=0,
                high=num_classes,
                size=(args.batch_size, ),
                device=dist_util.dev())
            model_kwargs['y'] = classes
        sample_fn = (
            diffusion.p_sample_loop
            if not args.use_ddim else diffusion.ddim_sample_loop)
        sample = sample_fn(
            model,
            (args.batch_size, 3, image_size, image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        if distributed:
            gathered_samples = [
                th.zeros_like(sample) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_samples,
                            sample)  # gather not supported with NCCL
            all_images.extend(
                [sample.cpu().numpy() for sample in gathered_samples])

            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend(
                [labels.cpu().numpy() for labels in gathered_labels])
        else:
            all_images.extend([s.cpu().numpy() for s in sample])
            all_labels.extend([lab.cpu().numpy() for lab in classes])

        if rank == 0:
            pbar.update(args.batch_size)
            logger.info(f'created {len(all_images) * args.batch_size} samples')

    arr = np.concatenate(all_images, axis=0)
    arr = arr[:args.num_samples]

    if num_classes > 0:
        if len(all_labels) != 1:
            label_arr = np.concatenate(all_labels, axis=0)
            label_arr = label_arr[:args.num_samples]
        else:
            label_arr = all_labels[0]

    if rank == 0:
        shape_str = 'x'.join([str(x) for x in arr.shape])
        out_path = os.path.join(cfg.work_dir, f'samples_{shape_str}.npz')
        logger.info(f'saving to {out_path}')
        if num_classes > 0:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    if distributed:
        dist.barrier()
    logger.info('sampling complete')


if __name__ == '__main__':
    main()
