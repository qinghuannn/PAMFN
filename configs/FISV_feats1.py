features = {"V": "VST", "F": "I3D", "A": "AST"}

single_modality = {
    "batch_size": 32,
    "epoch": {"TES": 250, "PCS": 250},
    "num_workers": 16,
    
    "model_dim": 256,
    "in_dim": {"V": 1024, "F": 1024, "A": 768},
    "drop_rate": 0.5,
    
    "optim": "SGD",
    "lr": 1e-2,
    "weight_decay": 1e-4,
    "reduce_fc_lr": False,
    
    "lr_scheduler": "cosine",
    "lr_min": 5e-6,
    "warmup_lr_init": 5e-6,    
    
    "clip_grad": True,
    "save_moniter": "avg_coef"
}
    
    
multi_modality = {
    "batch_size": 32,
    "epoch": {"TES": 500, "PCS": 500},
    "num_workers": 2,
    
    "model_dim": 256,
    "K": 6,
    "fc_drop": 0,
    "fc_r": 2,
    "feat_drop": 0.5,
    "ms_heads": 1,
    "cm_heads": 1,
    
    "ckpt_dir": "./pretrained_models/feats1",
    "rgb_ckpt_name": "VST",
    "flow_ckpt_name": "I3D",
    "audio_ckpt_name": "AST",
    
    "optim": "AdamW",
    "lr": 8e-4,
    "weight_decay": 1e-4,
    "reduce_fc_lr": True,

    "lr_scheduler": "none",
    "lr_min": 5e-6,
    "warmup_lr_init": 5e-6,    
    
    "clip_grad": True,
    "save_moniter": None,
}