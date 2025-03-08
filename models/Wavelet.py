import math
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1, Inception_Block_V2

import ptwt
import numpy as np

# 导入SWIFT模块接口
from swiftmodule.swift_modules import SelectiveTemporalStateSpace, MultiScaleDilatedConv, FeatureInteractionBridge, DynamicScaleSelection

try:
    from torch import irfft
    from torch import rfft
except ImportError:
    def rfft(x, d):
        t = torch.fft.fft(x, dim=(-d))
        r = torch.stack((t.real, t.imag), -1)
        return r

    def irfft(x, d):
        t = torch.fft.ifft(torch.complex(x[..., 0], x[..., 1]), dim=(-d))
        return t.real

def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = rfft(v, 1)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[..., 0] * W_r - Vc[..., 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V

class DCTChannelBlock(nn.Module):
    def __init__(self, channel, sequence_length):
        super(DCTChannelBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(channel, channel * 2, kernel_size=1, bias=False),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel * 2, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.dct_norm = nn.LayerNorm([sequence_length], eps=1e-6)

    def forward(self, x):
        b, c, l = x.size()  # (B, C, L)
        list_dct = []
        for i in range(c):
            freq = dct(x[:, i, :])
            list_dct.append(freq)
        stack_dct = torch.stack(list_dct, dim=1)
        
        lr_weight = self.dct_norm(stack_dct)
        lr_weight = self.fc(lr_weight)  # (B, C, L)
        
        return x * lr_weight  # (B, C, L)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

def Wavelet_for_Period(x, scale=16):
    scales = 2 ** np.linspace(-1, scale, 8)
    coeffs, freqs = ptwt.cwt(x, scales, "morl")
    return coeffs, freqs


class Wavelet(nn.Module):
    def __init__(self, configs):
        super(Wavelet, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.k = configs.top_k
        self.period_coeff = configs.period_coeff if hasattr(configs, 'period_coeff') else 0.5
        self.scale = configs.wavelet_scale

        if(configs.tib==1):
            self.period_conv = nn.Sequential(
                Inception_Block_V1(configs.d_model, configs.d_ff,
                                num_kernels=configs.num_kernels),
                nn.GELU(),
                Inception_Block_V1(configs.d_ff, configs.d_model,
                                num_kernels=configs.num_kernels),
            )
        elif(configs.tib==2):
            self.period_conv = nn.Sequential(
                Inception_Block_V2(configs.d_model, configs.d_ff,
                                num_kernels=configs.num_kernels),
                nn.GELU(),
                Inception_Block_V2(configs.d_ff, configs.d_model,
                                num_kernels=configs.num_kernels),
            )
        
        self.scale_conv = nn.Conv2d(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                kernel_size=(8, 1),
                stride=1,
                padding=(0, 0),
                groups=configs.d_model)
    
        self.projection = nn.Linear(self.seq_len + self.pred_len, self.pred_len, bias=True)
        self.dct_block = DCTChannelBlock(channel=configs.d_model, sequence_length=configs.seq_len + configs.pred_len)
        
        # SWIFT 模块接口
        self.use_swift_modules = False  # 控制是否启用SWIFT模块
        
        # 创建SWIFT模块接口
        self.stss = SelectiveTemporalStateSpace(
            d_model=configs.d_model, 
            seq_len=configs.seq_len + configs.pred_len
        )
        
        self.msdcn = MultiScaleDilatedConv(
            d_model=configs.d_model,
            num_scales=4  # 默认使用4个尺度
        )
        
        self.fib = FeatureInteractionBridge(
            d_model=configs.d_model
        )
        
        self.dss = DynamicScaleSelection(
            pred_len=configs.pred_len,
            num_scales=4
        )

    def forward(self, x):
        # 保存原始输入以备后用
        x_orig = x
        
        # FFT
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.period_conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
            
        if len(res) > 0:
            res = torch.stack(res, dim=-1)
            # adaptive aggregation
            period_weight = F.softmax(period_weight, dim=1)
            period_weight = period_weight.unsqueeze(
                1).unsqueeze(1).repeat(1, T, N, 1)

        # Wavelet
        coeffs = Wavelet_for_Period(x.permute(0, 2, 1), self.scale)[0].permute(1, 2, 0, 3).float()
        wavelet_res = self.period_conv(coeffs)
        wavelet_res = self.scale_conv(wavelet_res).squeeze(2).permute(0, 2, 1)

        if len(res) > 0:
            res = (1 - self.period_coeff ** 10) * wavelet_res + (self.period_coeff ** 10) * torch.sum(res * period_weight, -1)
        else:
            res = wavelet_res
            
        res = res + x
        
        # 使用SWIFT模块进行前向传播
        if self.use_swift_modules:
            # 使用时会激活这些模块的计算
            h_s = self.stss(res)
            h_c = self.msdcn(res)
            h_s, h_c = self.fib(h_s, h_c)
            
            # 获取小波能量用于动态尺度选择
            dummy_wavelet_energies = torch.ones(B, 4, device=x.device)
            res = self.dss(dummy_wavelet_energies, h_s, h_c)
        
        # 修改：插入 FECAM 模块
        res = res.permute(0, 2, 1)  # 调整形状为 (B, C, L)
        res = self.dct_block(res)
        res = res.permute(0, 2, 1)  # 调整回原形状

        res = res + x
        return self.projection(res.permute(0, 2, 1)).permute(0, 2, 1)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.layer = configs.e_layers
        self.wavelet_model = nn.ModuleList([Wavelet(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        
        # 控制是否启用SWIFT功能的参数
        self.use_swift = False

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # align temporal dimension

        # TimesNet
        for i in range(self.layer):
            # 如果启用SWIFT功能，设置wavelet模型中的use_swift_modules标志
            if self.use_swift:
                self.wavelet_model[i].use_swift_modules = True
            else:
                self.wavelet_model[i].use_swift_modules = False
                
            enc_out = self.layer_norm(self.wavelet_model[i](enc_out))

        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        if dec_out.shape[1] > self.pred_len:
            dec_out = dec_out * \
                    (stdev[:, 0, :].unsqueeze(1).repeat(
                        1, self.pred_len + self.seq_len, 1))
            dec_out = dec_out + \
                    (means[:, 0, :].unsqueeze(1).repeat(
                        1, self.pred_len + self.seq_len, 1))
        else:
            dec_out = dec_out * \
                    (stdev[:, 0, :].unsqueeze(1).repeat(
                        1, self.pred_len, 1))
            dec_out = dec_out + \
                    (means[:, 0, :].unsqueeze(1).repeat(
                        1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]  # [B, L, D]