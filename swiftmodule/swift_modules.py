import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pywt

# SWIFT模块简单实现，商业版本请洽谈

class SelectiveTemporalStateSpace(nn.Module):
    """
    Selective Temporal State Space (STSS) 模块
    扩展了Mamba与时间序列特定的门控机制
    """
    def __init__(self, d_model, seq_len, dropout=0.1):
        super(SelectiveTemporalStateSpace, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        # 框架接口
        self.layer_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, d_model * 3)  # 生成 ∆, B, C 参数
        self.temporal_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.gate_activation = nn.Sigmoid()
        
    def forward(self, x):
        return x


class MultiScaleDilatedConv(nn.Module):
    """
    Multi-Scale Dilated Convolutional Network (MSDCN)
    采用并行膨胀卷积与自适应感受野
    """
    def __init__(self, d_model, num_scales=4, dropout=0.1):
        super(MultiScaleDilatedConv, self).__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        
        # 框架接口
        self.wavelet_decomp = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, dilation=2**i),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=1)
            ) for i in range(num_scales)
        ])
        
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        
    def forward(self, x):
        return x


class FeatureInteractionBridge(nn.Module):
    """
    Feature Interaction Bridge (FIB)
    促进跨路径信息交换
    """
    def __init__(self, d_model, dropout=0.1):
        super(FeatureInteractionBridge, self).__init__()
        self.d_model = d_model
        
        # 框架接口
        self.q_s = nn.Linear(d_model, d_model)
        self.k_c = nn.Linear(d_model, d_model)
        self.v_c = nn.Linear(d_model, d_model)
        
        self.q_c = nn.Linear(d_model, d_model)
        self.k_s = nn.Linear(d_model, d_model)
        self.v_s = nn.Linear(d_model, d_model)
        
        self.gate_s_c = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        self.gate_c_s = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, h_s, h_c):
        return h_s, h_c


class DynamicScaleSelection(nn.Module):
    """
    Dynamic Scale Selection (DSS)
    基于预测范围的自适应时间尺度权重
    """
    def __init__(self, pred_len, num_scales=4):
        super(DynamicScaleSelection, self).__init__()
        self.pred_len = pred_len
        self.num_scales = num_scales
        
        # 框架接口
        self.alpha_params = nn.Parameter(torch.ones(num_scales))
        self.beta_params = nn.Parameter(torch.ones(num_scales) * pred_len / 2)
        
    def forward(self, wavelet_energies, h_s, h_c):
        alpha = 0.5
        beta = 0.2
        return alpha * h_s + (1 - alpha) * h_c + beta * (h_s * h_c)
