import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pywt


class SelectiveTemporalStateSpace(nn.Module):
    """
    Selective Temporal State Space (STSS) 模块
    扩展了Mamba与时间序列特定的门控机制
    """
    def __init__(self, d_model, seq_len, dropout=0.1):
        super(SelectiveTemporalStateSpace, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 参数生成投影
        self.projection = nn.Linear(d_model, d_model * 3)  # 生成 ∆, B, C 参数
        
        # 时间卷积进行上下文感知
        self.temporal_conv = nn.Conv1d(
            d_model, 
            d_model, 
            kernel_size=3, 
            padding=1,
            groups=d_model  # 深度可分离卷积
        )
        
        # 激活函数
        self.gate_activation = nn.Sigmoid()
        
        # S4D参数
        self.time_scale = nn.Parameter(torch.randn(d_model))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入序列，形状为 [B, L, D]
            
        Returns:
            torch.Tensor: 处理后的序列
        """
        # 保存输入用于残差连接
        residual = x
        
        # 层归一化
        x = self.layer_norm(x)
        
        # 生成SSM参数
        projection = self.projection(x)  # [B, L, 3D]
        delta, B, C = torch.split(projection, self.d_model, dim=-1)
        
        # 应用sigmoid激活确保delta为正
        delta = F.softplus(delta)
        
        # 时间门控 - 使用卷积获取上下文信息
        # 转置以使通道维度在中间
        x_conv = x.transpose(1, 2)  # [B, D, L]
        gate = self.temporal_conv(x_conv)  # [B, D, L]
        gate = gate.transpose(1, 2)  # [B, L, D]
        gate = self.gate_activation(gate)
        
        # 应用门控到SSM参数
        delta_gated = delta * gate
        B_gated = B * gate
        C_gated = C * gate
        
        # S4D状态空间计算
        batch_size, seq_len, hidden_dim = x.shape
        device = x.device
        
        # 初始化隐藏状态
        h = torch.zeros(batch_size, hidden_dim, device=device)
        
        # 离散化 SSM A 矩阵
        time_scale = torch.exp(self.time_scale)  # [D]
        A = torch.exp(-delta_gated * time_scale.unsqueeze(1))  # [B, L, D]
        
        outputs = []
        
        # 按序列长度逐步处理
        for t in range(seq_len):
            # 当前输入
            xt = x[:, t, :]  # [B, D]
            
            # 当前步的参数
            At = A[:, t, :]  # [B, D]
            Bt = B_gated[:, t, :]  # [B, D]
            Ct = C_gated[:, t, :]  # [B, D]
            
            # 状态更新
            h = At * h + Bt * xt
            
            # 输出计算
            yt = Ct * h  # [B, D]
            outputs.append(yt)
        
        # 堆叠输出
        output = torch.stack(outputs, dim=1)  # [B, L, D]
        
        # 残差连接
        output = residual + self.dropout(output)
        
        return output


class MultiScaleDilatedConv(nn.Module):
    """
    Multi-Scale Dilated Convolutional Network (MSDCN)
    采用并行膨胀卷积与自适应感受野
    """
    def __init__(self, d_model, num_scales=4, dropout=0.1):
        super(MultiScaleDilatedConv, self).__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        
        # 小波分解层 - 使用不同尺度的卷积
        self.wavelet_decomp = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=2**i, dilation=2**i),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=1)
            ) for i in range(num_scales)
        ])
        
        # 尺度权重 - 自适应加权
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        
        # 尺度特定的注意力
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Linear(d_model, d_model * num_scales),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入序列，形状为 [B, L, D]
            
        Returns:
            torch.Tensor: 处理后的序列
        """
        # 保存输入用于残差连接
        residual = x
        
        # 转置用于卷积
        x_conv = x.transpose(1, 2)  # [B, D, L]
        
        # 执行小波变换模拟 - 这里使用膨胀卷积模拟
        multi_scale_features = []
        for i, conv in enumerate(self.wavelet_decomp):
            # 应用膨胀卷积
            scale_feature = conv(x_conv)  # [B, D, L]
            multi_scale_features.append(scale_feature)
        
        # 计算动态尺度权重
        x_pool = x_conv.mean(dim=2, keepdim=True)  # [B, D, 1]
        scale_attn = self.scale_attention(x_conv)  # [B, D*num_scales, 1]
        scale_attn = scale_attn.view(x.size(0), self.d_model, self.num_scales, 1)  # [B, D, S, 1]
        scale_attn = scale_attn.mean(dim=1)  # [B, S, 1]
        
        # 计算尺度特定权重
        softmax_weights = F.softmax(self.scale_weights, dim=0)
        
        # 组合不同尺度的特征
        combined_feature = torch.zeros_like(multi_scale_features[0])
        for i, feature in enumerate(multi_scale_features):
            scale_weight = scale_attn[:, i, :] * softmax_weights[i]
            combined_feature += feature * scale_weight.unsqueeze(1)
        
        # 转置回原始形状
        output = combined_feature.transpose(1, 2)  # [B, L, D]
        
        # 残差连接
        output = residual + self.dropout(output)
        
        return output


class FeatureInteractionBridge(nn.Module):
    """
    Feature Interaction Bridge (FIB)
    促进跨路径信息交换
    """
    def __init__(self, d_model, dropout=0.1):
        super(FeatureInteractionBridge, self).__init__()
        self.d_model = d_model
        
        # 请求/键/值投影
        self.q_s = nn.Linear(d_model, d_model)
        self.k_c = nn.Linear(d_model, d_model)
        self.v_c = nn.Linear(d_model, d_model)
        
        self.q_c = nn.Linear(d_model, d_model)
        self.k_s = nn.Linear(d_model, d_model)
        self.v_s = nn.Linear(d_model, d_model)
        
        # 交互门控
        self.gate_s_c = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        self.gate_c_s = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, h_s, h_c):
        """
        促进STSS和MSDCN之间的交互
        
        Args:
            h_s (torch.Tensor): STSS路径特征，形状为 [B, L, D]
            h_c (torch.Tensor): MSDCN路径特征，形状为 [B, L, D]
            
        Returns:
            tuple: 交互后的STSS和MSDCN特征
        """
        batch_size, seq_len, dim = h_s.shape
        
        # STSS → MSDCN 交互
        # STSS特征作为查询，MSDCN特征作为键和值
        q_s = self.q_s(h_s)  # [B, L, D]
        k_c = self.k_c(h_c)  # [B, L, D]
        v_c = self.v_c(h_c)  # [B, L, D]
        
        # 注意力计算
        attn_s_c = torch.bmm(q_s, k_c.transpose(1, 2)) / math.sqrt(dim)  # [B, L, L]
        attn_s_c = F.softmax(attn_s_c, dim=2)
        
        # 应用注意力
        context_s_c = torch.bmm(attn_s_c, v_c)  # [B, L, D]
        
        # 门控机制
        gate_s_c = self.gate_s_c(h_s)
        h_s_updated = h_s + gate_s_c * context_s_c
        
        # MSDCN → STSS 交互
        # MSDCN特征作为查询，STSS特征作为键和值
        q_c = self.q_c(h_c)  # [B, L, D]
        k_s = self.k_s(h_s)  # [B, L, D]
        v_s = self.v_s(h_s)  # [B, L, D]
        
        # 注意力计算
        attn_c_s = torch.bmm(q_c, k_s.transpose(1, 2)) / math.sqrt(dim)  # [B, L, L]
        attn_c_s = F.softmax(attn_c_s, dim=2)
        
        # 应用注意力
        context_c_s = torch.bmm(attn_c_s, v_s)  # [B, L, D]
        
        # 门控机制
        gate_c_s = self.gate_c_s(h_c)
        h_c_updated = h_c + gate_c_s * context_c_s
        
        return h_s_updated, h_c_updated


class DynamicScaleSelection(nn.Module):
    """
    Dynamic Scale Selection (DSS)
    基于预测范围的自适应时间尺度权重
    """
    def __init__(self, pred_len, num_scales=4):
        super(DynamicScaleSelection, self).__init__()
        self.pred_len = pred_len
        self.num_scales = num_scales
        
        # 尺度特定的参数
        self.alpha_params = nn.Parameter(torch.ones(num_scales))
        self.beta_params = nn.Parameter(torch.ones(num_scales) * pred_len / 2)
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * num_scales, 2),
            nn.Softmax(dim=1)
        )
        
        # 尺度交互参数
        self.interaction_param = nn.Parameter(torch.tensor(0.2))
        
    def forward(self, wavelet_energies, h_s, h_c):
        """
        基于小波能量和预测范围动态选择时间尺度
        
        Args:
            wavelet_energies (torch.Tensor): 小波能量，形状为 [B, S]
            h_s (torch.Tensor): STSS路径特征，形状为 [B, L, D]
            h_c (torch.Tensor): MSDCN路径特征，形状为 [B, L, D]
            
        Returns:
            torch.Tensor: 融合后的特征
        """
        batch_size = h_s.shape[0]
        
        # 计算尺度权重 - 使用Sigmoid函数
        # 我们计算每个尺度的权重，基于1/((1 + exp(-(pred_len - beta) / alpha))) 公式
        pred_len_tensor = torch.full((batch_size, 1), self.pred_len, 
                                     device=h_s.device)
        
        # 为每个尺度计算权重
        scale_weights = []
        for i in range(self.num_scales):
            beta = self.beta_params[i]
            alpha = torch.abs(self.alpha_params[i]) + 1e-5  # 确保为正
            
            # 计算sigmoid - 更长的预测范围会增加低频权重
            weight = 1 / (1 + torch.exp(-(pred_len_tensor - beta) / alpha))
            scale_weights.append(weight)
        
        scale_weights = torch.cat(scale_weights, dim=1)  # [B, S]
        
        # 组合小波能量和尺度权重
        combined_weights = torch.cat([scale_weights, wavelet_energies], dim=1)  # [B, 2*S]
        
        # 预测STSS和MSDCN路径的融合权重
        fusion_weights = self.fusion_layer(combined_weights)  # [B, 2]
        
        # 提取alpha和beta权重
        alpha = fusion_weights[:, 0].unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
        beta = fusion_weights[:, 1].unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
        
        # 计算交互项
        interaction = self.interaction_param * (h_s * h_c)
        
        # 融合STSS和MSDCN特征
        fused_features = alpha * h_s + (1 - alpha) * h_c + beta * interaction
        
        return fused_features
