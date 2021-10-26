_base_ = ['base.py']

memcache_args = dict(
    backend='memcached',
    server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf',
    client_cfg='/mnt/lustre/share/memcached_client/client.conf',
    sys_path='/mnt/lustre/share/pymc/py3')

# 32 * 4 = 128
data = dict(
    type='RepeatDataset',
    times=1000,
    samples_per_gpu=8,
    workers_per_gpu=4,
    dataset=dict(
        data_dir='./data/cifar_train',
        image_size=32,
        memcache_args=memcache_args),
    persistent_workers=True,
)

model_and_diffusion = dict(
    learn_sigma=True,
    model=dict(
        image_size=32,
        num_channels=128,
        num_res_blocks=3,
        dropout=0,
    ),
    diffusion=dict(steps=4000, noise_schedule='cosine'))

train_cfg = dict(lr=1e-4, batch_size=128, save_dir='ckpt')

writer = dict(project='Improve-DDPM-CIFAR10', name='L_hybird')
