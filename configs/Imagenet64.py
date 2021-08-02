_base_ = ['base.py']

# MODEL_FLAGS="--image_size 64 --num_channels 192 --num_res_blocks 3 --learn_sigma True --class_cond True"  # noqa
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False"  # noqa
# TRAIN_FLAGS="--lr 3e-4 --batch_size 2048"

memcache_args = dict(
    backend='memcached',
    server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf',
    client_cfg='/mnt/lustre/share/memcached_client/client.conf',
    sys_path='/mnt/lustre/share/pymc/py3')

# bz = 256 * 8 = 2048
data = dict(
    data_dir='./data/imagenet',
    class_cond=True,
    image_size=64,
    memcache_args=memcache_args,
    samples_per_gpu=256,
    workers_per_gpu=5,
    persistent_workers=True,
)

model_and_diffusion = dict(
    num_classes=1000,
    learn_sigma=True,
    model=dict(
        image_size=64,
        num_channels=192,
        num_res_blocks=3,
        class_cond=True,
        dropout=0.3,
    ),
    diffusion=dict(
        steps=4000,
        noise_schedule='cosine',
        rescale_timesteps=False,
        rescale_learned_sigmas=False))

train_cfg = dict(lr=3e-4, batch_size=256, save_dir='ckpt')

writer = dict(project='Improve-DDPM', name='ImageNet-64-ClassCond')
