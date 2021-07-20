import logging
import os.path as osp
import subprocess
import sys
from collections import defaultdict

import cv2
import mmcv
import torch
import torchvision
from mmcv.runner import set_random_seed as set_random_seed_mmcv
from mmcv.utils import get_build_config, get_logger


def set_random_seed(seed, deterministic=False, use_rank_shift=True):
    """Set random seed.

    In this function, we just modify the default behavior of the simliar
    function defined in MMCV.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
        rank_shift (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Default: True.
    """
    set_random_seed_mmcv(
        seed, deterministic=deterministic, use_rank_shift=use_rank_shift)


def get_root_logger(log_file=None, log_level=logging.INFO, file_mode='w'):
    """Initialize and get a logger with name of mmgen.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger. Defaults to ``None``.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
            Defaults to ``logging.INFO``.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.

    Returns:
        logging.Logger: The expected logger.
    """
    return get_logger('mmgen', log_file, log_level, file_mode=file_mode)


def collect_env():
    """Collect the information of the running environments."""
    env_info = {}
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')

    cuda_available = torch.cuda.is_available()
    env_info['CUDA available'] = cuda_available

    if cuda_available:
        from mmcv.utils.parrots_wrapper import CUDA_HOME
        env_info['CUDA_HOME'] = CUDA_HOME

        if CUDA_HOME is not None and osp.isdir(CUDA_HOME):
            try:
                nvcc = osp.join(CUDA_HOME, 'bin/nvcc')
                nvcc = subprocess.check_output(
                    '"{}" -V | tail -n1'.format(nvcc), shell=True)
                nvcc = nvcc.decode('utf-8').strip()
            except subprocess.SubprocessError:
                nvcc = 'Not Available'
            env_info['NVCC'] = nvcc

        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for devname, devids in devices.items():
            env_info['GPU ' + ','.join(devids)] = devname

    gcc = subprocess.check_output('gcc --version | head -n1', shell=True)
    gcc = gcc.decode('utf-8').strip()
    env_info['GCC'] = gcc

    env_info['PyTorch'] = torch.__version__
    env_info['PyTorch compiling details'] = get_build_config()

    env_info['TorchVision'] = torchvision.__version__

    env_info['OpenCV'] = cv2.__version__

    env_info['MMCV'] = mmcv.__version__
    try:
        from mmcv.ops import get_compiler_version, get_compiling_cuda_version
        env_info['MMCV Compiler'] = get_compiler_version()
        env_info['MMCV CUDA Compiler'] = get_compiling_cuda_version()
    except ImportError:
        env_info['MMCV Compiler'] = 'n/a'
        env_info['MMCV CUDA Compiler'] = 'n/a'

    return env_info
