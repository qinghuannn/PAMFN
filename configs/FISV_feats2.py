features = {"V": "UMT", "F": "I3D", "A": "MAST"}

single_modality = {
    "batch_size": 32,
    "epoch": {"TES": 250, "PCS": 250},
    "num_workers": 2,
    
    "model_dim": 256,
    "in_dim": {"V": 1024, "F": 1024, "A": 768},
    "drop_rate": 0.3,
    
    "optim": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "reduce_fc_lr": False,
    
    "lr_scheduler": "none",
    "lr_min": 1e-6,
    "warmup_lr_init": 1e-6,    
    
    "clip_grad": False,
    "save_moniter": "avg_loss"
}
    
    
multi_modality = {
    "batch_size": 32,
    "epoch": {"TES": 100, "PCS": 100},
    "num_workers": 2,
    
    "model_dim": 256,
    "K": 8,
    "fc_drop": 0.3,
    "fc_r": 1,
    "feat_drop": 0.3,
    "ms_heads": 2,
    "cm_heads": 4,
    
    "ckpt_dir": "./pretrained_models/feats2/",
    "rgb_ckpt_name": "UMT",
    "flow_ckpt_name": "I3D",
    "audio_ckpt_name": "MAST",
    
    "optim": "AdamW",
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "reduce_fc_lr": False,

    "lr_scheduler": "none",
    "lr_min": 1e-6,
    "warmup_lr_init": 1e-6,    
    
    "clip_grad": False,
    "save_moniter": None,
}