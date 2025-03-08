import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding

from swiftmodule.swift_modules import SelectiveTemporalStateSpace, MultiScaleDilatedConv, FeatureInteractionBridge, DynamicScaleSelection


class Model(nn.Module):
    """
    SWIFT: State-space Wavelet Integrated Forecasting Technology
    结合选择性状态空间模型和多尺度小波分析的时间序列预测模型
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        
        # 控制是否使用SWIFT模块
        self.use_swift = configs.use_swift if hasattr(configs, 'use_swift') else True
        
        # SWIFT特定参数
        self.num_scales = configs.num_scales if hasattr(configs, 'num_scales') else 4
        self.fib_dropout = configs.fib_dropout if hasattr(configs, 'fib_dropout') else 0.1
        self.temporal_window = configs.temporal_window if hasattr(configs, 'temporal_window') else 3

        # 数据嵌入
        self.enc_embedding = DataEmbedding(
            configs.enc_in, 
            configs.d_model, 
            configs.embed, 
            configs.freq, 
            configs.dropout
        )
        
        # 如果使用SWIFT，创建SWIFT层
        if self.use_swift:
            self.layers = nn.ModuleList()
            for _ in range(configs.e_layers):
                # SWIFT组件
                stss = SelectiveTemporalStateSpace(
                    d_model=configs.d_model,
                    seq_len=self.seq_len + self.pred_len,
                    dropout=configs.dropout,
                    temporal_window=self.temporal_window
                )
                
                msdcn = MultiScaleDilatedConv(
                    d_model=configs.d_model,
                    num_scales=self.num_scales,
                    dropout=configs.dropout
                )
                
                fib = FeatureInteractionBridge(
                    d_model=configs.d_model,
                    dropout=self.fib_dropout
                )
                
                dss = DynamicScaleSelection(
                    d_model=configs.d_model,
                    pred_len=self.pred_len,
                    num_scales=self.num_scales
                )
                
                self.layers.append(nn.ModuleDict({
                    'stss': stss,
                    'msdcn': msdcn,
                    'fib': fib,
                    'dss': dss
                }))
        else:
            # 备选: 使用简单的前馈层作为后备
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(configs.d_model, configs.d_model),
                    nn.GELU(),
                    nn.Linear(configs.d_model, configs.d_model)
                ) for _ in range(configs.e_layers)
            ])

        # 层归一化和预测线性层
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def swift_forward(self, x):
        """SWIFT模型前向传播"""
        for layer in self.layers:
            # SWIFT特有的处理
            h_s = layer['stss'](x)
            h_c, wavelet_energies = layer['msdcn'](x)
            h_s, h_c = layer['fib'](h_s, h_c)
            x = layer['dss'](wavelet_energies, h_s, h_c)
        return x
    
    def simple_forward(self, x):
        """简单备用前向传播"""
        for layer in self.layers:
            x = layer(x) + x  # 残差连接
        return x

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 归一化 (从Non-stationary Transformer)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # 嵌入
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # 对齐时间维度

        # 选择使用SWIFT或简单处理
        if self.use_swift:
            enc_out = self.swift_forward(enc_out)
        else:
            enc_out = self.simple_forward(enc_out)
            
        # 层归一化
        enc_out = self.layer_norm(enc_out)

        # 预测输出
        dec_out = self.projection(enc_out)

        # 反归一化 (从Non-stationary Transformer)
        if dec_out.shape[1] > self.pred_len:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        else:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]