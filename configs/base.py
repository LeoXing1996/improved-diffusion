# log_cfg = dict(log_interval=10,)
# save_cfg = dict(save_interval=10000)

train_cfg = dict(
    schedule_sampler='uniform',
    microbatch=-1,  # -1 disables microbatches
    lr=1e-4,
    weight_decay=0.0,
    lr_anneal_steps=0,
    use_fp16=False,
    fp16_scale_growth=1e-3,
    ema_rate='0.9999',  # comma-separated list of EMA values
    # tmp put them here
    log_interval=10,
    save_interval=10000)

model_and_diffusion = dict(
    learn_sigma=False,
    model=dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions='16,8',
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,  # gradient checkpoints
        use_scale_shift_norm=True,  # what this actually mean?
    ),
    diffusion=dict(
        steps=1000,
        sigma_small=True,
        noise_schedule='linear',
        timestep_respacing='',
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
    ))

dist_params = dict(backend='nccl')
