"""Generate a large batch of image samples from a model and save them as a
large numpy array.

This can be used to produce samples for FID evaluation.
"""

import argparse
import os
from copy import deepcopy

import numpy as np
import torch as th
import torch.distributed as dist
from improved_diffusion import dist_util, logger
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (NUM_CLASSES, add_dict_to_argparser,
                                            create_gaussian_diffusion,
                                            create_model)
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist
from torch.nn.parallel.distributed import DistributedDataParallel


def main():
    args = create_argparser().parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, backend='nccl')
        # re-set gpu_ids with distributed training mode
    rank, world_size = get_dist_info()

    logger.configure()

    logger.log('creating model and diffusion...')

    cfg = Config.fromfile(args.config)

    learn_sigma = cfg.model_and_diffusion.get('learn_sigma', False)
    model_cfg_ = deepcopy(cfg.model_and_diffusion.model)
    model_cfg_['learn_sigma'] = learn_sigma
    diffusion_cfg_ = deepcopy(cfg.model_and_diffusion.diffusion)
    diffusion_cfg_['learn_sigma'] = learn_sigma

    train_cfg_ = deepcopy(cfg.train_cfg)

    model = create_model(**model_cfg_)

    diffusion = create_gaussian_diffusion(**diffusion_cfg_)
    sampler = create_named_schedule_sampler(
        train_cfg_.get('schedule_sampler', 'uniform'), diffusion)
    train_cfg_['schedule_sampler'] = sampler

    model.load_state_dict(th.load(args.ckpt, map_location='cpu'))
    if distributed:
        model = DistributedDataParallel(
            model.cuda(),
            device_ids=[th.cuda.current_device()],
            find_unused_parameters=True)
    else:
        model.cuda()
    model.eval()

    image_size = cfg.model_and_diffusion.model['image_size']

    logger.log('sampling...')
    all_images = []
    all_labels = []

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0,
                high=NUM_CLASSES,
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

        gathered_samples = [
            th.zeros_like(sample) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_samples,
                        sample)  # gather not supported with NCCL
        all_images.extend(
            [sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend(
                [labels.cpu().numpy() for labels in gathered_labels])
        if rank == 0:
            logger.log(f'created {len(all_images) * args.batch_size} samples')

    arr = np.concatenate(all_images, axis=0)
    arr = arr[:args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[:args.num_samples]
    if dist.get_rank() == 0:
        shape_str = 'x'.join([str(x) for x in arr.shape])
        out_path = os.path.join(args.work_dir, f'samples_{shape_str}.npz')
        logger.log(f'saving to {out_path}')
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log('sampling complete')


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=16,
        use_ddim=False,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('ckpt', type=str)
    parser.add_argument('--class_cond', default=None)
    parser.add_argument('--work-dir', type=str, default='work_dirs')
    parser.add_argument('--num-samples', type=int, default=10000)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == '__main__':
    main()
