
import os
import os.path as osp

import torch

import numpy as np
from torch.utils.data import Dataset


def zero_padding_torch(data, target_length):
    original_length = data.shape[0]
    zero_data = torch.zeros((target_length-original_length, ) + data.shape[1:])
    return torch.cat([data, zero_data], dim=0)


def get_default_segments(dataset):
    if dataset in ['Ball', 'Clubs', 'Hoop', 'Ribbon']:
        return 70
    elif dataset in ['TES', 'PCS']:
        return 130
    else:
        print("action error!")
        exit(0)


class AQADataset(Dataset):
    def __init__(self, dataset_name, feat_dir,
                 rgb_feat, flow_feat, audio_feat,
                 squeeze_rgb_feat=None, squeeze_flow_feat=None,
                 is_train=True, segments=-1, **kwargs):
        super().__init__()
        self.is_train = is_train
        self.squeeze_rgb_feat = squeeze_rgb_feat
        self.squeeze_flow_feat = squeeze_flow_feat
        self.labels = self.build(dataset_name, is_train)
        self.datas = self.read_data(dataset_name, feat_dir, rgb_feat, flow_feat, audio_feat)
        if segments < 0:
            self.segments = get_default_segments(dataset_name)
        else:
            self.segments = segments

    def build(self, dataset_name, is_train):
        labels = []
        if dataset_name in ['Ball', 'Clubs', 'Hoop', 'Ribbon']:
            label_path = osp.join("./data/RG/", "train.txt" if is_train else "test.txt")
            with open(label_path, 'r') as fr:
                for i, line in enumerate(fr.readlines()):
                    if i == 0:
                        continue
                    line = line.strip().split()
                    if line[0].startswith(dataset_name):
                        labels.append((line[0], float(line[3]) / 25))
        elif dataset_name in ["TES", "PCS"]:
            label_path = osp.join("./data/FISV", "train.txt" if is_train else "test.txt")
            with open(label_path, 'r') as fr:
                for i, line in enumerate(fr.readlines()):
                    line = line.strip().split()
                    if dataset_name == 'TES':
                        labels.append((line[0], float(line[1]) / 45))
                    elif dataset_name == 'PCS':
                        labels.append((line[0], float(line[2]) / 45))
        else:
            raise ValueError(f"{dataset_name} not supported!")
        return labels

    def read_data(self, dataset_name, feat_dir, rgb_feat, flow_feat, audio_feat):
        if dataset_name in ["TES", "PCS"]:
            dataset_name = "FISV"
        rgb_data = np.load(osp.join(feat_dir, f"{dataset_name}_rgb_{rgb_feat}.npy"), allow_pickle=True).item()
        flow_data = np.load(osp.join(feat_dir, f"{dataset_name}_flow_{flow_feat}.npy"), allow_pickle=True).item()
        audio_data = np.load(osp.join(feat_dir, f"{dataset_name}_audio_{audio_feat}.npy"), allow_pickle=True).item()
        return {"rgb": rgb_data, "flow": flow_data, "audio": audio_data}

    def squeeze_dim(self, rgb_data, flow_data, audio_data):
        if self.squeeze_rgb_feat == 'mean' and len(rgb_data.shape) == 3:
            rgb_data = rgb_data.mean(dim=1)
        elif self.squeeze_rgb_feat == 'cat' and len(rgb_data.shape) == 3:
            rgb_data = rgb_data.view(rgb_data.shape[0], -1)
        if self.squeeze_flow_feat and len(flow_data.shape) == 3:
            flow_data = flow_data.mean(dim=1)
        elif self.squeeze_flow_feat and len(flow_data) == 3:
            flow_data = flow_data.view(flow_data.shape[0], -1)
        return rgb_data, flow_data, audio_data

    def __getitem__(self, idx):
        sample_name, label = self.labels[idx]
        rgb_data = torch.from_numpy(self.datas['rgb'][sample_name]).float()
        flow_data = torch.from_numpy(self.datas['flow'][sample_name]).float()
        audio_data = torch.from_numpy(self.datas['audio'][sample_name]).float()

        rgb_data, flow_data, audio_data = self.squeeze_dim(rgb_data, flow_data, audio_data)

        if self.is_train:
            if len(rgb_data) > self.segments:
                start_idx = np.random.randint(len(rgb_data) - self.segments)
                rgb_data = rgb_data[start_idx:start_idx + self.segments]
                audio_data = audio_data[start_idx:start_idx + self.segments]
                flow_data = flow_data[start_idx:start_idx + self.segments]
            elif len(rgb_data) < self.segments:
                rgb_data = zero_padding_torch(rgb_data, self.segments)
                audio_data = zero_padding_torch(audio_data, self.segments)
                flow_data = zero_padding_torch(flow_data, self.segments)
        # print(rgb_data.shape)
        # return rgb_data, flow_data, audio_data, torch.tensor(label, dtype=torch.float)
        # return rgb_data, flow_data, audio_data, label
        return rgb_data, audio_data, flow_data, label, 0

    def __len__(self):
        return len(self.labels)

