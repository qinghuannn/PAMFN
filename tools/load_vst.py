import os
import sys
user_home = os.path.expanduser('~')
vst_repo_home = os.path.join(user_home, '/path/Video-Swin-Transformer')
sys.path.append(vst_repo_home)

import mmcv
import torch
from mmcv.runner import load_checkpoint
from mmaction.models import build_recognizer


def load_vst(config, checkpoint):
    config = mmcv.Config.fromfile(config)
    config.model.backbone.pretrained = None
    model = build_recognizer(config.model, test_cfg=config.get('test_cfg'))
    model.cfg = config
    load_checkpoint(model, checkpoint, map_location='cpu')
    model = torch.nn.Sequential(
        model.backbone,
        torch.nn.AdaptiveAvgPool3d((1, 1, 1))
    )
    return model