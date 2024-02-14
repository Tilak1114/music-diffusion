data_config = {
    "train_file_path":"/data/tilak/projects/mustango/data/MusicBench_train.json",
    "val_file_path":"/data/tilak/projects/mustango/data/MusicBench_testB.json",
    "test_file_path": "/data/tilak/projects/mustango/data/MusicBench_testA.json",
}

pretrained_configs = {
    "text_encoder":"google/flan-t5-large",
    "audio_ldm":"audioldm-s-full",
    "ddpm_scheduler":"stabilityai/stable-diffusion-2-1"
}

module_config = {
    "learning_rate":4.5e-5,
    "weight_decay":1e-8,
    "gradient_accumulation_steps": 4,
    "lr_scheduler":"linear",
    "adam_beta1":0.9,
    "adam_beta2":0.999,
    "adam_weight_decay":1e-2,
    "adam_eps":1e-08,
}

trainer_config = {
    "max_epochs":-1,
    "devices":-1
}

