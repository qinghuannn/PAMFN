import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

from einops import rearrange


def smart_fn(fn, feat_list, func_list=False):
    if func_list is False:
        return [fn(x) for x in feat_list]
    return [fn[idx](x) for idx, x in enumerate(feat_list)]


class BaseConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.GELU()
        )

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class BaseModel(nn.Module):
    def __init__(self, in_dim, model_dim, drop_rate, modality="V"):
        super().__init__()
        dim = model_dim
        self.modality = modality
        self.embedding = nn.Sequential(
            nn.Conv1d(in_dim, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
            nn.Dropout(0.3)
        )
        self.stage1 = BaseConvBlock(dim)
        self.stage2 = BaseConvBlock(dim)
        self.stage3 = BaseConvBlock(dim)
        self.pool = nn.AvgPool1d(2, 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        # self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Conv1d(dim, 1, 1),
            nn.Sigmoid()
        )
        self.mse = nn.MSELoss()

    def forward(self, feats):
        x = feats[self.modality]
        if len(x.shape) == 4:
            x = x.mean(dim=2)
        x = rearrange(x, 'b t d -> b d t').contiguous()
        x = self.embedding(x)

        x1 = self.stage1(x)
        x1 = self.pool(x1)

        x2 = self.stage2(x1)
        x2 = self.pool(x2)

        x3 = self.stage3(x2)
        x3 = self.gap(x3)

        score = self.fc(x3).squeeze(dim=2)
        return score, {'feats': [x, x1, x2, x3]}

    def call_loss(self, pred, label, **kwargs):
        return self.mse(pred.squeeze(), label.squeeze())


class Multi_Head_Attention(nn.Module):
    def __init__(self, num_heads, dim_model, dropout=0.1):
        super(Multi_Head_Attention, self).__init__()
        self.num_heads = num_heads
        assert dim_model % num_heads == 0
        self.dim_head = dim_model // self.num_heads
        self.linear_layers = nn.ModuleList([nn.Linear(dim_model, dim_model) for _ in range(3)])
        self.drop = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(dim_model)

    def self_attention(self, Q, K, V, mask=None, dropout=None):
        scores = -1 * torch.matmul(Q, K.transpose(-2, -1)) \
                 / np.sqrt(Q.size(-1))
        if mask is not None:
            # scores = scores.masked_fill(mask == 0, -1e9)
            mask = mask.unsqueeze(dim=1).repeat([1, self.num_heads, 1, 1])
            # scores = torch.mul(scores, mask)  # mask
            scores = scores + mask  # mask
        attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            attn = dropout(attn)
        context = torch.matmul(attn, V)
        return context, attn

    def forward(self, input_Q, input_K, input_V, mask=None):
        residual_Q, b = input_Q, input_V.size(0)
        Q, K, V = [l(x).view(b, -1, self.num_heads, self.dim_head).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (input_Q, input_K, input_V))]
        x, attn = self.self_attention(Q, K, V, mask=mask, dropout=self.drop)
        x = x.transpose(1, 2).contiguous().view(b, -1, self.num_heads * self.dim_head)
        # x = self.norm(x + residual_Q)
        return x


class MS_Fusion(nn.Module):
    def __init__(self, dim, num_heads=1, dropout=0.1):
        super().__init__()
        self.fusion_s = Multi_Head_Attention(num_heads=num_heads, dim_model=dim, dropout=dropout)
        self.proj = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
        )

    def forward(self, s, c):
        q = rearrange(self.proj(c), 'b c t -> (b t) 1 c')
        s = torch.stack(s, dim=2)
        s = rearrange(s, 'b c n t -> (b t) n c')
        ns = self.fusion_s(q, s, s)
        ns = rearrange(ns, '(b t) 1 c -> b c t', b=c.shape[0])
        return ns


class CM_Fusion(nn.Module):
    def __init__(self, dim, K,  num_heads=1,  gap=False, dropout=0.1,):
        super().__init__()
        self.K = K
        self.proj1 = nn.ModuleList([nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
            nn.Conv1d(dim, self.K, 1),
        ) for _ in range(3)
        ])
        self.ffn = nn.ModuleList([nn.Sequential(
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True)
        ) for _ in range(self.K)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1) if gap else nn.AvgPool1d(2, 2)

        self.fusion_f = Multi_Head_Attention(num_heads=num_heads, dim_model=dim, dropout=dropout)
        self.proj2 = nn.Sequential(
            nn.Conv1d(2 * dim, dim, 1)
        )

        self.policy = nn.Sequential(
            nn.Conv1d(3*dim, K, 1)
        )

    def gen_mask(self, fusion_kind):
        b, k, t = fusion_kind.shape
        mask = torch.zeros_like(fusion_kind, device=fusion_kind.device)
        # b k t
        idx = torch.arange(0, k, device=fusion_kind.device).repeat([b, 1, 1]).permute([0, 2, 1]).repeat([1, 1, t])
        pos_mask = idx > torch.mul(fusion_kind, idx).sum(dim=1, keepdim=True) # b 1 t
        mask[pos_mask] = -1e9
        mask = mask + fusion_kind - fusion_kind.detach() # b k t
        return mask

    def gen_fusion_kind(self, f, tau):
        # B 3C T -> B N T
        logit = self.pool(self.policy(torch.cat(f, dim=1)))
        fusion_kind = F.gumbel_softmax(logit, tau, hard=True, dim=1)
        mask = self.gen_mask(fusion_kind)
        return fusion_kind, mask

    def forward(self, f, c, s, tau):
        att = torch.softmax(torch.stack(smart_fn(self.proj1, f, True), dim=1), dim=1)
        att = [_ for _ in att.split(1, dim=2)]
        nf = [torch.mul(_, torch.stack(f, dim=1)).sum(dim=1) for _ in att]
        nf = torch.stack([self.pool(self.ffn[idx](_) + _) for idx, _ in enumerate(nf)], dim=2)
        nf = rearrange(nf, 'b c n t -> (b t) n c')

        q = self.proj2(torch.cat([c, s], dim=1))
        q = rearrange(q, 'b c t -> (b t) 1 c')
        q = -1 * q
        fusion_kind, mask = self.gen_fusion_kind(f, tau)
        mask = rearrange(mask, 'b n t -> (b t) 1 n')
        nf = self.fusion_f(q, nf, nf, mask=mask)
        nf = rearrange(nf, '(b t) 1 c -> b c t', b=c.shape[0])
        return nf, fusion_kind


class FusionBlock(nn.Module):
    def __init__(self, dim, K, ms_heads, cm_heads, gap=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim*3, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
            nn.Conv1d(dim, dim, 3, 1, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True)
        )
        self.short_cut = nn.Sequential(
            nn.Conv1d(dim*3, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(True)
        )
        self.s_fusion = MS_Fusion(dim, ms_heads)
        self.f_fusion = CM_Fusion(dim, K, cm_heads, gap)

    def forward(self, c, f, s, tau):
        c = self.conv(c) + self.short_cut(c)
        s = self.s_fusion(s, c)
        f, action = self.f_fusion(f, c, s, tau)
        c = torch.cat([c, s, f], dim=1)
        return c, action

class PAMFN(nn.Module):
    def __init__(self,
                 model_dim, fc_drop, fc_r, feat_drop, K,
                 ms_heads, cm_heads,
                 ckpt_dir, rgb_ckpt_name, flow_ckpt_name, audio_ckpt_name,
                 dataset_name):
        super().__init__()
        self.model_r = torch.load(osp.join(ckpt_dir, f"{dataset_name}_rgb_{rgb_ckpt_name}.pth"), map_location='cpu')
        self.model_f = torch.load(osp.join(ckpt_dir, f"{dataset_name}_flow_{flow_ckpt_name}.pth"), map_location='cpu')
        self.model_a = torch.load(osp.join(ckpt_dir, f"{dataset_name}_audio_{audio_ckpt_name}.pth"), map_location='cpu')

        self.c = nn.Parameter(torch.zeros([model_dim*3]), requires_grad=False)
        self.stage1 = FusionBlock(model_dim, K, ms_heads, cm_heads)
        self.stage2 = FusionBlock(model_dim, K, ms_heads, cm_heads)
        self.stage3 = FusionBlock(model_dim, K, ms_heads, cm_heads, True) 
        self.pool = nn.AvgPool1d(2, 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(feat_drop)

        self.fc = nn.Sequential(
            nn.Dropout(fc_drop),
            nn.Conv1d(model_dim * 3, model_dim//fc_r, 1),
            nn.BatchNorm1d(model_dim//fc_r),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(model_dim//fc_r, 1, 1),
            nn.Sigmoid()
        )
        self.tau = nn.Parameter(torch.ones([]) * 10)
        self.mse = nn.MSELoss()

    def forward(self, input_feats):
        _, feat_r = self.model_r(input_feats)
        _, feat_a = self.model_a(input_feats)
        _, feat_f = self.model_f(input_feats)
                
        feats = [[self.drop(_.detach()) for _ in feat_r['feats']],
                 [self.drop(_.detach()) for _ in feat_f['feats']],
                 [self.drop(_) for _ in feat_a['feats']]]

        x = self.c.repeat([feats[0][0].shape[0], feats[0][0].shape[2]//2, 1]).permute([0, 2, 1])

        x1, action1 = self.stage1(x, [_[0] for _ in feats], [_[1] for _ in feats], self.tau)
        x1 = self.pool(x1)

        x2, action2 = self.stage2(x1, [_[1] for _ in feats], [_[2] for _ in feats], self.tau)
        x2 = self.gap(x2)

        x3, action3 = self.stage3(x2, [_[2] for _ in feats], [_[3] for _ in feats], self.tau)

        score = self.fc(x3).squeeze(dim=2)
        return score, {'action': [action1, action2, action3]}

    def update(self, **kwargs):
        return

    def call_loss(self, pred, label, **kwargs):
        return self.mse(pred.squeeze(), label.squeeze())