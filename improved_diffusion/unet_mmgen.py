# Copyright (c) OpenMMLab. All rights reserved.
import sys

from mmgen.models.builder import build_module  # noqa

# yapf: disable
sys.path.append('/space0/home/xingzn/mmgen_dev/DDPM')  # isort:skip  # noqa
# yapf: enable


def get_mmgen_denoising():
    denoising = dict(
        type='DenoisingUnet',
        image_size=32,
        in_channels=3,
        base_channels=128,
        resblocks_per_downsample=3,
        attention_res=[16, 8],
        use_scale_shift_norm=True,
        dropout=0,
        num_heads=4,
        rescale_timesteps=False,
        output_cfg=dict(mean='eps', var='learned_range'),
        style='official')
    return build_module(denoising)


if __name__ == '__main__':
    get_mmgen_denoising()
