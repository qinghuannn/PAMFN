from typing import List, Tuple

import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from datasets import AQADataset as Dset
from utils.utils import Logger, AverageMeter, now_to_date
from utils.lr_scheduler import build_scheduler


def build_dataset(dataset_name, feat_dir, rgb_feat, flow_feat, audio_feat,
                  batch_size=32, num_workers=2, segments=-1):
    train_dset = Dset(dataset_name=dataset_name, feat_dir=feat_dir,
                      rgb_feat=rgb_feat, flow_feat=flow_feat, audio_feat=audio_feat,
                      squeeze_rgb_feat="mean",
                      segments=segments, is_train=True)
    test_dset = Dset(dataset_name=dataset_name, feat_dir=feat_dir,
                     rgb_feat=rgb_feat, flow_feat=flow_feat, audio_feat=audio_feat,
                     squeeze_rgb_feat="mean",
                     segments=segments, is_train=False)
    train_dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=True,
                               pin_memory=False, num_workers=num_workers)
    test_dloader = DataLoader(test_dset, batch_size=1, shuffle=False,
                              pin_memory=False, num_workers=num_workers)
    return train_dloader, test_dloader
 

def build_model(model_name, model_conf=None):
    print(model_conf)
    if model_name == "pamfn_base":
        from models.pamfn import BaseModel
        return BaseModel(**model_conf)
    elif model_name == "pamfn_final":
        from models.pamfn import PAMFN
        return PAMFN(**model_conf)
    else:
        raise ValueError(f"model: {model_name} is not supported!")


def build_optimizer(model, optim, lr, momentum=0.9, weight_decay=None, reduce_fc_lr=False, model_name=None):
    if reduce_fc_lr and model_name == 'pamfn_final':
        lr_scale = 0.1
        fc_params = list(map(id, model.fc.parameters()))
        other_params = filter(lambda p: id(p) not in fc_params, model.parameters())
        params_group = [
            {'params': model.fc.parameters(), 'lr': lr * lr_scale},
            {'params': other_params}
        ]
    else:
        params_group = model.parameters()
    if optim == 'Adam':
        optimizer = torch.optim.Adam(params_group, lr=lr)
    elif optim == 'SGD':
        optimizer = torch.optim.SGD(params_group, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim == 'AdamW':
        optimizer = torch.optim.AdamW(params_group, lr=lr)
    else:
        raise ValueError(f"optimizer: {optim} not found!")
    return optimizer


def build_lr_scheduler(optimizer, scheduler, epoch_num, scheduler_steps,
                    lr_min=5e-6, warmup_lr_init=5e-6, decay_rate=0.1):
    return build_scheduler(optimizer, scheduler, epoch_num, scheduler_steps, decay_rate=decay_rate,
                    lr_min=lr_min, warmup_lr_init=warmup_lr_init)


def train_epoch(model, optimizer, lr_scheduler, dataloader, cur_epoch, clip_grad=False):
    losses = AverageMeter()
    total_preds = []
    total_labels = []
    tm = time.time()

    model.train()

    for i, (video_data, audio_data, flow_data, label, _) in enumerate(dataloader):
        video_data = video_data.cuda()
        audio_data = audio_data.cuda()
        flow_data = flow_data.cuda()
        label = label.float().cuda()

        preds, other_info = model({"V": video_data, "F": flow_data, "A": audio_data})
        loss = model.call_loss(preds, label, **other_info)

        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()

        losses.update(loss.item(), len(label))
        total_preds.extend(preds.cpu().detach().numpy().tolist())
        total_labels.extend(label.cpu().detach().numpy().tolist())

    if lr_scheduler is not None:
        lr_scheduler.step(cur_epoch+1)

    tm = time.time() - tm
    coef, _ = spearmanr(total_preds, total_labels)

    return tm, losses.avg * 100, coef


def test_epoch(model, dataloader):
    losses = AverageMeter()
    total_preds = []
    total_labels = []
    tm = time.time()

    model.eval()

    with torch.no_grad():
        for i, (video_data, audio_data, flow_data, label, _) in enumerate(dataloader):
            video_data = video_data.cuda()
            audio_data = audio_data.cuda()
            flow_data = flow_data.cuda()
            label = label.float().cuda()

            preds, other_info = model({"V": video_data, "F": flow_data, "A": audio_data})
            loss = model.call_loss(preds, label, **other_info)

            losses.update(loss.item(), label.shape[0])
            total_preds.extend(preds.cpu().detach().numpy().tolist())
            total_labels.extend(label.cpu().detach().numpy().tolist())


    if type(total_preds[0]) is List:
        total_preds = [x[0] for x in total_preds]
    coef, _ = spearmanr(total_preds, total_labels)

    tm = time.time() - tm

    return tm, losses.avg * 100, coef * 100


def train_loop(model, optimizer, lr_scheduler, train_dataloader, test_dataloader, epoch_num,
               save_base_model="", save_final_model="", save_moniter="",
                log_func=None, clip_grad=False):
    max_coef, max_avg_coef, min_loss, min_avg_loss = 0, 0, 10, 10
    coef_of_saved_model = 0
    _train_loss, _test_loss = [], []
    train_coefs, test_coefs = [], []
    for cur_epoch in range(epoch_num):
        train_tm, train_loss, train_coef = train_epoch(model, optimizer, lr_scheduler, train_dataloader,
                                                       cur_epoch, clip_grad=clip_grad)
        test_tm, test_loss, test_coef = test_epoch(model, test_dataloader)

        train_coefs.append(train_coef), test_coefs.append(test_coef)
        _train_loss.append(train_loss), _test_loss.append(test_loss) 
        if save_base_model != '':
            if save_moniter == "avg_coef" and np.mean(test_coefs[-10:]) > max_avg_coef: 
                torch.save(model, save_base_model)
                coef_of_saved_model = test_coef
            if save_moniter == "avg_loss" and np.mean(_test_loss[-10:]) > min_avg_loss:
                torch.save(model, save_base_model)
                coef_of_saved_model = test_coef
        if save_final_model != "" and test_coef > max_coef:
            torch.save(model.state_dict(), save_final_model)

        max_coef = max(max_coef, test_coef)
        max_avg_coef = max(max_avg_coef, np.mean(test_coefs[-10:]))
        min_loss = min(min_loss, test_loss)
        min_avg_loss = min(min_avg_loss, np.mean(_test_loss[-10:]))
        log_func('Epoch[{0}/{1}] \t'
                 'Time: {train_tm:.1f}/{test_tm:.1f} \t'
                 'Loss {train_loss:.4f}/{test_loss:.4f} \t'
                 'Avg_loss {avg_loss:.2f}/{min_avg_loss:.2f}\t'
                 'Coef {train_coef:.2f}/{test_coef:.2f} \t'
                 'Avg_coef {avg_coef:.2f}/{max_avg_coef:.2f} \t'
                 'BestCoef {max_coef:.2f}\t BestLoss {min_loss:.2f}\t '.format(
            cur_epoch, epoch_num, train_tm=train_tm, test_tm=test_tm,
            train_loss=train_loss, test_loss=test_loss,
            avg_loss=np.mean(_test_loss[-10:]), min_avg_loss=min_avg_loss,
            train_coef=train_coef, test_coef=test_coef,
            avg_coef=np.mean(test_coefs[-10:]), max_avg_coef=max_avg_coef,
            max_coef=max_coef, min_loss=min_loss
        ))
    return max_avg_coef, max_coef, min_avg_loss, min_loss, coef_of_saved_model


def go(model_kwargs, optimizer_kwargs, lr_scheduler_kwargs, dataset_kwargs, epoch_num,
         save_base_model="", save_final_model="", save_moniter="",
         log_func=None, seed=0, clip_grad=False):
    setup_seed(seed)
    train_dataloader, test_dataloader = build_dataset(**dataset_kwargs)

    model = build_model(**model_kwargs)
    optimizer = build_optimizer(model, model_name=model_kwargs["model_name"], **optimizer_kwargs)
    lr_scheduler = build_lr_scheduler(optimizer, **lr_scheduler_kwargs)

    model.cuda()
    rets = train_loop(model, optimizer, lr_scheduler, train_dataloader, test_dataloader, epoch_num=epoch_num, 
            save_base_model=save_base_model, save_moniter=save_moniter, save_final_model=save_final_model, 
            log_func=log_func, clip_grad=clip_grad)
    log_func(f"best avg coef: {rets[0]/100:.4f}\tbest coef:{rets[1]/100:.4f}\t"
             f"min avg loss: {rets[2]/100:.6f}\tmin loss:{rets[3]/100:.6f}\t"
             f"coef of saved model: {rets[4]/100:.4f}")
    return rets


def print_flops(model_kwargs):
    from thop import clever_format, profile
    model = build_model(**model_kwargs)
    data_in = {
        "V": torch.zeros([1, 70, 1024]),
        "F": torch.zeros([1, 70, 1024]),
        "A": torch.zeros([1, 70, 768])
    }
    macs, params = profile(model, inputs=(data_in,))
    macs, params = clever_format([macs, params], "%.3f")
    return macs, params


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  


def log_info(info, logger, print_log):
    if logger is not None:
        logger.info(info)
    elif print_log is not False:
        print(info)

