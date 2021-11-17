_base_ = ['base.py']

# MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True"  # noqa
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"  # noqa
# TRAIN_FLAGS="--lr 1e-4 --batch_size 128"  # noqa

file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        '.data/imagenet/':
        'openmmlab:s3://openmmlab/classification/imagenet/',
    }))

# bz = 16 * 8 = 128
data = dict(
    data_dir='./data/imagenet',
    image_size=64,
    file_client_args=file_client_args,
    samples_per_gpu=16,
    workers_per_gpu=5,
    persistent_workers=True,
)

model_and_diffusion = dict(
    learn_sigma=True,
    model=dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=3,
    ),
    diffusion=dict(
        steps=4000,
        noise_schedule='cosine',
    ))

# train for 1500k
train_cfg = dict(lr=1e-4, save_dir='ckpt', max_iterations=1500000)

writer = dict(project='Improve-DDPM', name='ImageNet-64-ClassCond')
