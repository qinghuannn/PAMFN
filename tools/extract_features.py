# -*- coding: utf-8 -*-
import argparse
import os
import time
import os.path as osp
import cv2
import librosa
import torch
import torchaudio
import torchvision
import torchaudio.transforms as atf
import torchvision.transforms as vtf
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator

from glob import glob
from tqdm import tqdm, trange
from multiprocessing.pool import ThreadPool
import threading

from torchlibrosa.stft import Spectrogram, LogmelFilterBank

import sys
sys.path.append('./')

import utils.video_transforms as video_transforms
from models.i3d import I3D
from models.audio_model import ResNet22, Cnn14, Cnn14_16k
from models.AST import ASTModel
from tools.load_vst import load_vst

# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)

user_home = os.path.expanduser('~')


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model_name, modality, audio_tdim=0):
    model_name = modality + '_' + model_name
    if model_name == 'rgb_I3D':
        check_point = user_home + '/datas/pretrained_models/i3d_rgb.pth'
        model = I3D(400, modality='rgb')
        model.load_state_dict(torch.load(check_point, map_location='cpu'))
    elif model_name == 'rgb_VST':
        # config = user_home + '/repos/ActionRecognition/Video-Swin-Transformer/'\
        #                      'configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k.py'
        # check_point = user_home + '/datas/pretrained_models/swin_base_patch244_window877_kinetics400_22k.pth'
        config = user_home + '/repos/action_recognition/Video-Swin-Transformer/'\
                             'configs/recognition/swin/swin_base_patch244_window877_kinetics600_22k.py'
        check_point = user_home + '/datas/pretrained_models/vst_swin_base_patch244_window877_kinetics600_22k.pth'
        model = load_vst(config, check_point)
    elif model_name == 'flow_I3D':
        check_point = user_home + '/datas/pretrained_models/i3d_flow.pth'
        model = I3D(400, modality='flow')
        model.load_state_dict(torch.load(check_point, map_location='cpu'))
    elif model_name == 'audio_PANNs_ResNet22':
        check_point = user_home + '/datas/pretrained_models/panns_resnet22.pth'
        model = ResNet22(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
                         fmin=50, fmax=14000, classes_num=527)
        model.load_state_dict(torch.load(check_point, map_location='cpu')['models'])
    elif model_name == 'audio_PANNs_CNN14':
        check_point = user_home + '/datas/pretrained_models/panns_Cnn14_mAP=0.431.pth'
        model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
                      fmin=50, fmax=14000, classes_num=527)
        model.load_state_dict(torch.load(check_point, map_location='cpu')['models'])
    elif model_name == 'audio_PANNs_CNN14_16k':
        check_point = user_home + '/datas/pretrained_models/panns_Cnn14_16k_mAP=0.438.pth'
        model = Cnn14_16k(sample_rate=16000, window_size=512, hop_size=160, mel_bins=64,
                      fmin=50, fmax=8000, classes_num=527)
        model.load_state_dict(torch.load(check_point, map_location='cpu')['models'])
    elif model_name == 'audio_AST':
        check_point = user_home + '/datas/pretrained_models/vst_audioset_10_10_0.4593.pth'
        model = ASTModel(input_tdim=audio_tdim, audioset_pretrain=True, pretrain_model_path=check_point)
    else:
        raise ValueError('models name %s incorrect!' % model_name)
    model.eval()
    model.cuda()
    return model


def build_transform(model_name, modality):
    model_name = modality + '_' + model_name
    if model_name == 'rgb_I3D':
        transform = vtf.Compose([
            vtf.Resize(256),
            vtf.CenterCrop((224, 300)),
            vtf.Resize((224, 224)),
            vtf.ToTensor(),
            vtf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
        ])
    elif model_name == 'flow_I3D':
        transform = vtf.Compose([
            vtf.Resize(256),
            vtf.CenterCrop((224, 300)),
            vtf.Resize((224, 224)),
            vtf.ToTensor(),
            vtf.Normalize(mean=[0.5], std=[0.5]) 
        ])
    elif model_name == 'rgb_VST':
        transform = vtf.Compose([
            vtf.Resize(256),
            vtf.CenterCrop((224, 300)),
            vtf.Resize((224, 224)),
            vtf.ToTensor(),
            vtf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError('models name %s incorrect!' % model_name)
    return transform


# Extract features from data after frame extraction and optical flow generation by densflow.
# Since the number of frames extracted by densflow differs from that extracted by OpenCV, it is listed separately.
class Dset_AVF(Dataset):
    def __init__(self, dataset, video_path, optical_path,
                 audio_path, rgb_transform=None, flow_transform=None,
                frame_sample_rate=1, frame_per_segment=32, audio_model='PANNs',
                target_modality='AVF', tm_dim=84, audio_sr=32000):
        self.video_info = sorted([(osp.join(video_path, _),
                                   osp.join(optical_path, _),
                                   osp.join(audio_path, _+'.wav'))
                           for _ in os.listdir(video_path)])[250:]
        self.dataset = dataset
        self.target_modality = target_modality
        self.frame_stride = frame_sample_rate
        self.frame_per_segment = frame_per_segment

        self.audio_model = audio_model
        self.audio_sample_rate = audio_sr
        self.tm_dim = tm_dim

        self.rgb_transform = rgb_transform
        self.flow_transform = flow_transform

    def read_audio(self, audio_path, num_segments):
        if self.audio_model[:5] == 'PANNs':
            wave, _ = librosa.load(audio_path, sr=self.audio_sample_rate, mono=True)
            sample_points_per_segment = len(wave) // num_segments
            # PANNs uses wave as input.
            audio_data = np.array(wave[:sample_points_per_segment * num_segments], dtype=np.float32)
            audio_data = audio_data.reshape([num_segments, -1])   # [num_seg, t]
            audio_data = torch.from_numpy(audio_data)
        elif self.audio_model == 'AST':
            wave, sr = torchaudio.load(audio_path)
            wave = torchaudio.transforms.Resample(sr, new_freq=self.audio_sample_rate)(wave)
            wave = wave - wave.mean()
            sample_points_per_segment = wave.shape[1] // num_segments
            fbank = [torchaudio.compliance.kaldi.fbank(wave[:, i*sample_points_per_segment:(i+1)*sample_points_per_segment],
                                                       htk_compat=True, sample_frequency=sr, use_energy=False,
                                                       window_type='hanning', num_mel_bins=128, dither=0.0,
                                                       frame_shift=10) for i in range(num_segments)]
            fbank = (torch.stack(fbank) - 4.26) / (4.57 * 2)   # normalize by mean and std in audioset.
            p = self.tm_dim - fbank.shape[1]
            if p > 0:
                fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p), "constant", 0)
            else:
                fbank = fbank[:, :self.tm_dim]
            audio_data = fbank  # [num_seg, time_len, mel_bins]

        else:
            raise ValueError("audio models name %s incorrect!" % self.audio_model)
        return audio_data

    def read_video(self, rgb_path, flow_path, audio_path):
        rgb_frames_path = sorted(glob(osp.join(rgb_path, '*.jpg')))
        flow_frames_path = sorted(glob(osp.join(flow_path, '*.jpg')))

        if len(rgb_frames_path) * 2 - 2 != len(flow_frames_path):   # for debug
            print(rgb_path, flow_path)
            print(len(rgb_frames_path), len(flow_frames_path))
        assert len(rgb_frames_path) * 2 - 2 == len(flow_frames_path)

        num_rgb_frames = len(rgb_frames_path)
        rgb_frames, flow_frames = [], []
        rgb_data, flow_data, audio_data = [], [], []
        # the number of final sampled frames：len(rgb_frames_path) // self.frame_stride
        num_segments = len(rgb_frames_path) // self.frame_stride // self.frame_per_segment

        # read rgb frame
        if 'V' in self.target_modality:
            for idx in range(num_rgb_frames):
                if idx % self.frame_stride == 0:
                        frame = cv2.imread(rgb_frames_path[idx])
                        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        frame = self.rgb_transform(frame)
                        rgb_frames.append(frame)
            rgb_data = torch.stack(rgb_frames[:num_segments * self.frame_per_segment], dim=0)
            # 32 frame per segment
            rgb_data = rgb_data.view(num_segments, -1, rgb_data.shape[1], rgb_data.shape[2], rgb_data.shape[3])
            # 16 frame per segment. After extracting the features, convert two video segments to correspond to one audio segment.
            # rgb_data = rgb_data.view(num_segments*2, -1, rgb_data.shape[1], rgb_data.shape[2], rgb_data.shape[3])
            rgb_data = rgb_data.permute([0, 2, 1, 3, 4])
        # read optical flow
        if 'F' in self.target_modality:
            for idx in range(num_rgb_frames):
                if idx % self.frame_stride == 0:
                    # due to flow frames num == rgb frames num - 1
                    flow_img_idx = idx if idx != num_rgb_frames - 1 else idx - 1
                    flow_x = cv2.imread(flow_frames_path[flow_img_idx], cv2.IMREAD_GRAYSCALE)
                    flow_y = cv2.imread(flow_frames_path[flow_img_idx + num_rgb_frames - 1], cv2.IMREAD_GRAYSCALE)
                    flow_x = self.flow_transform(Image.fromarray(flow_x))
                    flow_y = self.flow_transform(Image.fromarray(flow_y))
                    flow_frames.append(torch.cat([flow_x, flow_y], dim=0))
            flow_data = torch.stack(flow_frames[:num_segments * self.frame_per_segment], dim=0)
            # 32 frame per segment
            flow_data = flow_data.view(num_segments, -1, flow_data.shape[1], flow_data.shape[2], flow_data.shape[3])
            # 16 frame per segment. After extracting the features, convert two video segments to correspond to one audio segment.
            # flow_data = flow_data.view(num_segments * 2, -1, flow_data.shape[1], flow_data.shape[2], flow_data.shape[3])
            flow_data = flow_data.permute([0, 2, 1, 3, 4])
        # read audio
        if 'A' in self.target_modality:
            # Each video segment corresponds to one audio segment.
            audio_data = self.read_audio(audio_path, num_segments)
        return rgb_data, flow_data, audio_data

    def __getitem__(self, idx):
        rgb_path, flow_path, audio_path = self.video_info[idx]
        rgb_data, flow_data, audio_data = self.read_video(rgb_path, flow_path, audio_path)
        video_name = osp.basename(self.video_info[idx][0]).split('.')[0]
        return video_name, rgb_data, flow_data, audio_data

    def __len__(self):
        return len(self.video_info)


def extract_AVF_features(root_path, dataset, save_path,
                         stride=1, frame_per_segment=32,
                        rgb_model='I3D', flow_model='I3D', audio_model='PANNs',
                        target_modality='AVF', mini_batch=16, tm_dim=0, audio_sr=0):
    video_path = osp.join(root_path, 'optical_flow_rgb_frames')
    audio_path = osp.join(root_path, 'audios')
    flow_path = osp.join(root_path, 'optical_flow')

    rgb_transform = None
    flow_transform = None
    if 'V' in target_modality:
        model_rgb = load_model(rgb_model, 'rgb')
        rgb_transform = build_transform(rgb_model, 'rgb')
    if 'F' in target_modality:
        model_flow = load_model(flow_model, 'flow')
        flow_transform = build_transform(flow_model, 'flow')
    if 'A' in target_modality:
        model_audio = load_model(audio_model, 'audio', tm_dim)

    dset = Dset_AVF(dataset, video_path, flow_path, audio_path,
                    rgb_transform, flow_transform,
                    stride, frame_per_segment, audio_model=audio_model, target_modality=target_modality,
                    tm_dim=tm_dim, audio_sr=audio_sr)
    dloader = DataLoader(dset, batch_size=1, shuffle=False,
                         pin_memory=False, num_workers=8)
    for idx, rgb_data, flow_data, audio_data in tqdm(dloader):
        with torch.no_grad():
            rgb_feat, flow_feat, audio_feat = [], [], []
            if 'V' in target_modality:
                for j in range(0, rgb_data.shape[1], mini_batch):
                    end = min(j+mini_batch, rgb_data.shape[1])
                    data = rgb_data[0][j:end].cuda()
                    if rgb_model == 'VST':
                        feat = model_rgb(data).squeeze()
                    else:
                        feat = model_rgb(data, 'avg').squeeze()
                    if len(feat) != end - j:
                        feat = feat.unsqueeze(dim=0)
                    rgb_feat.append(feat.cpu())
                rgb_feat = torch.vstack(rgb_feat)
                # rgb_feat = rgb_feat.view((-1, 2, feat.shape[-1])) 
                # when split a 32-frame segment into two 16-frame segments
                np.save(osp.join(save_path, '%s_video' % idx[0]), rgb_feat)
            if 'F' in target_modality:
                for j in range(0, flow_data.shape[1], mini_batch):
                    end = min(j + mini_batch, flow_data.shape[1])
                    data = flow_data[0][j:end].cuda()
                    feat = model_flow(data, 'avg').squeeze()
                    if len(feat) != end - j:
                        feat = feat.unsqueeze(dim=0)
                    feat = feat.mean(dim=2)     # when a 32-frame segment is enabled
                    flow_feat.append(feat.cpu())
                flow_feat = torch.vstack(flow_feat)
                # flow_feat = flow_feat.view([-1, 2, feat.shape[-1]]) # when split a 32-frame segment into two 16-frame segments
                np.save(osp.join(save_path, '%s_flow' % idx[0]), flow_feat)
            if 'A' in target_modality:
                for j in range(0, audio_data.shape[1], mini_batch):
                    end = min(j+mini_batch, audio_data.shape[1])
                    data = audio_data[0][j:end].cuda()
                    feat = model_audio(data, stop='avg')
                    audio_feat.append(feat.cpu())
                audio_feat = torch.vstack(audio_feat)
                np.save(osp.join(save_path, '%s_audio' % idx[0]), audio_feat)

def main():
    setup_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat', type=str, default='AVF', required=True)
    parser.add_argument('--dataset', type=str, default='RG', required=True)
    parser.add_argument('--save_path', type=str, required=True)

    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--rgb_model', type=str, default='I3D')
    parser.add_argument('--flow_model', type=str, default='I3D')
    parser.add_argument('--audio_model', type=str, default='PANNs')

    parser.add_argument('--tm_dim', type=int, default=-1)
    parser.add_argument('--audio_sr', type=int, default=32000)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if args.dataset == 'RG':
        root_path = user_home + '/datas/AQA/RG_public/'
    elif args.dataset == 'MS':
        root_path = user_home + '/datas/AQA/MIT_AQA/MIT_Skating/'
    elif args.dataset == 'FISV':
        root_path = user_home + '/datas/AQA/FISV/'
    else:
        raise Exception('dataset not found!')

    if not osp.exists(args.save_path):
        os.mkdir(args.save_path)

    extract_AVF_features(root_path, args.dataset, args.save_path, stride=args.stride,
                         rgb_model=args.rgb_model, flow_model=args.flow_model, audio_model=args.audio_model,
                         target_modality=args.feat, mini_batch=args.batch_size,
                         tm_dim=args.tm_dim, audio_sr=args.audio_sr)


if __name__ == '__main__':
    main()
