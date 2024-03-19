import os
import json
import argparse
import numpy as np
from functools import partial
import importlib.util


from pipeline import go, log_info


def load_module_from_file(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', required=True)

    parser.add_argument('--feats', type=int, default=1, required=True)
    parser.add_argument('--action', type=str, default='Ball', required=True)
    parser.add_argument('--multi_modality', action='store_true', default=False)
    parser.add_argument('--modality', type=str, default='V')
    
    parser.add_argument('--feat_dir', type=str, default='./data/features')
    parser.add_argument('--check_path', type=str, default='./pretrained_models/')
    
    parser.add_argument('--save_base_model', type=str, default="")
    
    parser.add_argument('--print_flops', action='store_true', default=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_args()

    if args.gpu != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    if args.action in ["Ball", "Clubs", "Hoop", "Ribbon"]:
        if args.feats == 1:
            import configs.RG_feats1 as conf
        elif args.feats == 2:
            import configs.RG_feats2 as conf
        else:
            raise ValueError(f"feats-{args.feats} is not supported!")
    elif args.action in ["TES", "PCS"]:
        if args.feats == 1:
            import configs.FISV_feats1 as conf
        elif args.feats == 2:
            import configs.FISV_feats2 as conf
        else:
            raise ValueError(f"feats-{args.feats} is not supported!")


    if args.multi_modality:
        cur_conf = conf.multi_modality
    else:
        cur_conf = conf.single_modality

    dataset_config = {
        "dataset_name": args.action,
        "feat_dir": args.feat_dir,
        "rgb_feat": conf.features["V"],
        "flow_feat": conf.features["F"],
        "audio_feat": conf.features["A"],
        "batch_size": cur_conf["batch_size"],
        "num_workers": cur_conf["num_workers"],
    }

    if not args.multi_modality:
        model_config = {
            "model_name": "pamfn_base",
            "model_conf": {
                "in_dim": cur_conf["in_dim"][args.modality],
                "model_dim": cur_conf["model_dim"],
                "drop_rate": cur_conf["drop_rate"],
                "modality": args.modality
            }
        }
    else:
        model_config = {
            "model_name": "pamfn_final",
            "model_conf": {
                "model_dim": cur_conf["model_dim"],
                "fc_drop": cur_conf["fc_drop"],
                "fc_r": cur_conf["fc_r"],
                "feat_drop": cur_conf["feat_drop"],
                "K": cur_conf["K"],
                "ms_heads": cur_conf["ms_heads"],
                "cm_heads": cur_conf["cm_heads"],
                "sg_ckpt_dir": cur_conf["sg_ckpt_dir"], 
                "rgb_ckpt_name": cur_conf["rgb_ckpt_name"], 
                "flow_ckpt_name": cur_conf["flow_ckpt_name"],
                "audio_ckpt_name": cur_conf["audio_ckpt_name"],
                "dataset_name": args.action
            }
        }

    optimizer_config = {
        "optim": cur_conf["optim"],
        "lr": cur_conf["lr"],
        "weight_decay": cur_conf["weight_decay"],
        "reduce_fc_lr": cur_conf["reduce_fc_lr"]
    }

    lr_scheduler = {
        "scheduler": cur_conf["lr_scheduler"],
        "epoch_num": cur_conf["epoch"][args.action],
        "scheduler_steps": cur_conf["epoch"],
        "lr_min": cur_conf["lr_min"],
        "warmup_lr_init": cur_conf["warmup_lr_init"],
        "decay_rate": 0.1
    }

    if args.print_flops:
        from pipeline import print_flops
        macs, params = print_flops(model_config)
        print(macs)
        print(params)
        exit()

    log_func = partial(log_info, logger=None, print_log=True)

    ret = go(model_config, optimizer_config, lr_scheduler, dataset_config, cur_conf["epoch"][args.action],
        save_base_model=args.save_base_model, save_moniter=cur_conf["save_moniter"],
        log_func=log_func, seed=0, clip_grad=cur_conf["clip_grad"])
    print("%.4f\t%.4f\t%.4f\t%.4f" % (ret[0], ret[1], ret[2], ret[3]))
