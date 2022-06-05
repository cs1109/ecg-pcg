from telnetlib import GA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
import scipy.signal as signal
import librosa
from config import *
import numpy as np
from collections import OrderedDict

def get_same_padding(kernel_width, stride=1, n_feature=1):
    return ((n_feature-1)*stride-n_feature+kernel_width)//2

def conv_size(input_size, kernel_size=4, stride=2, padding=1):
    return (input_size-kernel_size+2*padding)/stride+1

class ResUnit(nn.Module):
    def __init__(self, n_samples_in, n_filters_in, n_samples_out, n_filters_out,
                 dropout_keep_prob=0.8, kernel_size=17,
                 postactivation_bn=False, activation_function=nn.ReLU):
        super().__init__()
        self.n_samples_in = n_samples_in
        self.downsample = n_samples_in // n_samples_out
        self.n_filters_in = n_filters_in
        self.n_samples_out = n_samples_out
        self.n_filters_out = n_filters_out
        self.dropout_rate = 1 - dropout_keep_prob
        self.kernel_size = kernel_size
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation_function
        self.skip_connection = self._skip_connection(self.downsample, self.n_filters_in)
        self.layer_1st = nn.Sequential(
            nn.Conv1d(self.n_filters_in, self.n_filters_out, self.kernel_size, padding=get_same_padding(self.kernel_size), bias=False),
            self._batch_norm_plus_activation(),
            self._drop_out(),
        )
        self.layer_2nd_pre = nn.Conv1d(self.n_filters_out, self.n_filters_out, self.kernel_size, padding=get_same_padding(self.kernel_size), bias=False, stride=self.downsample)
        self.layer_2nd_pos = nn.Sequential(
            self._batch_norm_plus_activation(),
            self._drop_out(),
        )

    def _drop_out(self):
        return nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()

    def _skip_connection(self, downsample, n_filters_in):
        model_list = []
        print(downsample)
        if downsample > 1:
            model_list.append(nn.MaxPool1d(downsample))
        elif downsample == 1:
            model_list.append(nn.Identity())
        else:
            raise ValueError("Number of samples should always decrease.")
        if n_filters_in != self.n_filters_out:
            model_list.append(nn.Conv1d(n_filters_in, self.n_filters_out, 1, bias=False))
        return nn.Sequential(*model_list)

    def _batch_norm_plus_activation(self):
        model_list = []
        if self.postactivation_bn:
            model_list.append(self.activation_function())
            model_list.append(nn.BatchNorm1d(self.n_filters_out))
        else:
            model_list.append(nn.BatchNorm1d(self.n_filters_out))
            model_list.append(self.activation_function())
        return nn.Sequential(*model_list)

    def forward(self, inputs):
        x, y = inputs
        y = self.skip_connection(y)
        x = self.layer_1st(x)
        x = self.layer_2nd_pre(x)
        x = x+y
        y = x
        x = self.layer_2nd_pos(x)
        return [x, y]

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = self.dropout(F.softmax(attn, dim=-1))
        attn = attn.sum(2).unsqueeze(3)/attn.shape[2]
        # attn = torch.softmax(attn*10, dim=2)
        output = (attn*v).sum(2)/attn.shape[2]
        # output = torch.matmul(attn, v)
        return output, attn.sum(1)[:,:,0]/attn.shape[1]

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q):
        k = q
        v = q
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = v.sum(1)/v.shape[1]
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        v, attn = self.attention(q, k, v)
        v = v.transpose(1, 2).contiguous().view(sz_b, -1)
        v = self.dropout(self.fc(v))
        v += residual
        v = self.layer_norm(v)
        return v, attn

class ResNetBackBone(nn.Module):
    def __init__(self, n_samples_in, n_filters_in, n_samples_out, n_filters_out):
        super().__init__()
        kernel_size = 17
        model_list = []
        model_list.append(nn.Conv1d(n_filters_in, 16, kernel_size, padding=get_same_padding(kernel_size), bias=False))
        model_list.append(nn.BatchNorm1d(16))
        model_list.append(nn.ReLU())
        self.m1 = nn.Sequential(*model_list)
        model_list = []
        model_list.append(ResUnit(n_samples_in, 16, n_samples_in//4, 32, kernel_size=kernel_size))
        # model_list.append(ResUnit(n_samples_in//4, 32, n_samples_in//20, 64, kernel_size=kernel_size))
        # model_list.append(ResUnit(n_samples_in//20, 64, n_samples_out, n_filters_out, kernel_size=kernel_size))
        model_list.append(ResUnit(n_samples_in//4, 32, n_samples_out, n_filters_out, kernel_size=kernel_size))
        self.m2 = nn.Sequential(*model_list)

    def forward(self, x):
        x = self.m1(x)
        x, _ = self.m2([x, x])
        return x


class ECGResNetBackBone(nn.Module):
    def __init__(self, in_channel=1):
        super().__init__()
        kernel_size = 17
        model_list = []
        model_list.append(nn.Conv1d(in_channel, 16, kernel_size, padding=get_same_padding(kernel_size), bias=False))
        model_list.append(nn.BatchNorm1d(16))
        model_list.append(nn.ReLU())
        self.m1 = nn.Sequential(*model_list)
        model_list = []
        # model_list.append(ResUnit(8000, 16, 2000, 16, kernel_size=kernel_size))
        model_list.append(ResUnit(2000, 16, 400, 32, kernel_size=kernel_size))
        model_list.append(ResUnit(400, 32, dim_out, 1, kernel_size=kernel_size))
        self.m2 = nn.Sequential(*model_list)

    def forward(self, x):
        x = self.m1(x)
        x, _ = self.m2([x, x])
        return x

class PCGResNetBackBone(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 17
        model_list = []
        model_list.append(nn.Conv1d(1, 16, kernel_size, padding=get_same_padding(kernel_size), bias=False))
        model_list.append(nn.BatchNorm1d(16))
        model_list.append(nn.ReLU())
        self.m1 = nn.Sequential(*model_list)
        model_list = []
        # model_list.append(ResUnit(8000, 16, 2000, 16, kernel_size=kernel_size))
        model_list.append(ResUnit(2000, 16, 400, 32, kernel_size=kernel_size))
        model_list.append(ResUnit(400, 32, dim_out, 1, kernel_size=kernel_size))
        self.m2 = nn.Sequential(*model_list)

    def forward(self, x):
        x = self.m1(x)
        x, _ = self.m2([x, x])
        return x

class PCGCLFResNetBackBone(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 17
        model_list = []
        model_list.append(nn.Conv1d(1, 16, kernel_size, padding=get_same_padding(kernel_size), bias=False))
        model_list.append(nn.BatchNorm1d(16))
        model_list.append(nn.ReLU())
        self.m1 = nn.Sequential(*model_list)
        model_list = []
        model_list.append(ResUnit(8000, 16, 2000, 16, kernel_size=kernel_size))
        model_list.append(ResUnit(2000, 16, 400, 32, kernel_size=kernel_size))
        model_list.append(ResUnit(400, 32, dim_out, 1, kernel_size=kernel_size))
        self.m2 = nn.Sequential(*model_list)

    def forward(self, x):
        x = self.m1(x)
        x, _ = self.m2([x, x])
        return x

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class ECGResCLF(nn.Module):
    def __init__(self, n_classes, n_head=None, in_channel=1):
        super().__init__()
        self.backbone = nn.Sequential(
            ECGResNetBackBone(in_channel=in_channel),
            nn.Flatten(start_dim=1),
        )
        self.clf = nn.Sequential(
            nn.Flatten(),
            Linear(dim_out, n_classes),
            nn.Softmax(dim=1)
        )
        self.clf_loss = nn.MSELoss()

    def forward(self, ecg, pcg, y_true):
        out = self.backbone(ecg)
        y_pred = self.clf(out)
        loss = self.clf_loss(y_true, y_pred)
        return y_pred, loss, torch.zeros(1)

class PCGResCLF(nn.Module):
    def __init__(self, n_classes, n_head=None):
        super().__init__()
        self.backbone = nn.Sequential(
            PCGCLFResNetBackBone(),
            nn.Flatten(start_dim=1),
        )
        self.clf = nn.Sequential(
            nn.Flatten(),
            Linear(dim_out, n_classes),
            nn.Softmax(dim=1)
        )
        self.clf_loss = nn.MSELoss()

    def forward(self, ecg, pcg, y_true):
        out = self.backbone(pcg)
        y_pred = self.clf(out)
        loss = self.clf_loss(y_true, y_pred)
        return y_pred, loss, torch.zeros(1)

class RTNet(nn.Module):
    def __init__(self, n_classes, n_head=20):
        super().__init__()
        self.ecg_backbone = nn.Sequential(
            ECGResNetBackBone(),
            nn.Flatten(start_dim=1),
        )
        self.pcg_backbone = nn.Sequential(
            PCGCLFResNetBackBone(),
            nn.Flatten(start_dim=1),
        )
        self.pcg_ecg_attention = nn.Sequential(
            MultiHeadAttention(n_head, dim_out, dim_query, dim_query)
        )
        self.clf = nn.Sequential(
            nn.Flatten(),
            Linear(dim_out, n_classes),
            nn.Softmax(dim=1)
        )
        self.clf_loss = nn.MSELoss()

    def forward(self, ecg, pcg, y_true):
        ecg_fea = self.ecg_backbone(ecg)
        pcg_fea = self.pcg_backbone(pcg)
        ecg_pcg_fea, w_pcg_ecg = self.pcg_ecg_attention(torch.stack([pcg_fea, ecg_fea]).transpose(1,0))
        y_pred = self.clf(ecg_pcg_fea)
        loss = self.clf_loss(y_true, y_pred)
        return y_pred, loss, w_pcg_ecg

class ConvBlock(nn.Module):
    def __init__(self, n_samples_in, n_filters_in, n_samples_out, n_filters_out,
                 dropout_keep_prob=0.8, kernel_size=17,
                 postactivation_bn=False, activation_function=nn.ReLU):
        super().__init__()
        self.n_samples_in = n_samples_in
        self.downsample = n_samples_in // n_samples_out
        self.n_filters_in = n_filters_in
        self.n_samples_out = n_samples_out
        self.n_filters_out = n_filters_out
        self.dropout_rate = 1 - dropout_keep_prob
        self.kernel_size = kernel_size
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation_function
        self.conv = nn.Sequential(
            nn.Conv1d(self.n_filters_in, self.n_filters_out, self.kernel_size, padding=get_same_padding(self.kernel_size), bias=False, stride=self.downsample),
            self._batch_norm_plus_activation(),
            self._drop_out(),
        )

    def _drop_out(self):
        return nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()

    def _batch_norm_plus_activation(self):
        model_list = []
        if self.postactivation_bn:
            model_list.append(self.activation_function())
            model_list.append(nn.BatchNorm1d(self.n_filters_out))
        else:
            model_list.append(nn.BatchNorm1d(self.n_filters_out))
            model_list.append(self.activation_function())
        return nn.Sequential(*model_list)

    def forward(self, x):
        return self.conv(x)

class ConvBlock2d(nn.Module):
    def __init__(self, n_samples_in, n_filters_in, n_samples_out, n_filters_out,
                 dropout_keep_prob=0.8, kernel_size=3,
                 postactivation_bn=False, activation_function=nn.ReLU):
        super().__init__()
        self.n_samples_in = n_samples_in
        self.downsample = n_samples_in // n_samples_out
        self.n_filters_in = n_filters_in
        self.n_samples_out = n_samples_out
        self.n_filters_out = n_filters_out
        self.dropout_rate = 1 - dropout_keep_prob
        self.kernel_size = kernel_size
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation_function
        self.conv = nn.Sequential(
            nn.Conv2d(self.n_filters_in, self.n_filters_out, self.kernel_size, padding=get_same_padding(self.kernel_size), bias=False, stride=self.downsample),
            self._batch_norm_plus_activation(),
            self._drop_out(),
        )

    def _drop_out(self):
        return nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()

    def _batch_norm_plus_activation(self):
        model_list = []
        if self.postactivation_bn:
            model_list.append(self.activation_function())
            model_list.append(nn.BatchNorm2d(self.n_filters_out))
        else:
            model_list.append(nn.BatchNorm2d(self.n_filters_out))
            model_list.append(self.activation_function())
        return nn.Sequential(*model_list)

    def forward(self, x):
        return self.conv(x)

def st(x):
    # S变换。输入为numpy的实矩阵，输出为numpy的复矩阵
    # 不会用可以问我
    H = np.fft.fft(x)
    n=len(x)
    t=np.append(np.arange(np.ceil(n/2)),np.arange(-np.floor(n/2),0))
    t2=np.reciprocal(t[1:])[None]
    t=t[None].T
    t3=np.matmul(t, t2)
    t4=np.exp(-2*np.pi*np.pi*np.power(t3,2))
    t5=np.zeros([n,1])
    t5[0]=1
    t6=np.append(t5,t4,axis=1)
    t7=H[None]
    tt=np.arange(0,n)
    for i in range(1,n):
        t7=np.append(t7,H[np.roll(tt,-i)][None],axis=0)
    return np.fft.fft(np.fft.ifft2(t6*t7)).T

npst = np.frompyfunc(st, 1, 1)

class HanLiNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        kernel_size = 17
        self.ecg_conv = nn.Conv1d(1, 16, kernel_size, padding=get_same_padding(kernel_size), bias=False)
        self.pcg_conv = nn.Conv1d(1, 16, kernel_size, padding=get_same_padding(kernel_size), bias=False)
        self.raw_net = nn.Sequential(
            ConvBlock(4000, 16, 2000, 32, kernel_size=kernel_size),
            ConvBlock(2000, 32, 1000, 32, kernel_size=kernel_size),
            ConvBlock(1000, 32, 500, 32, kernel_size=kernel_size),
            ConvBlock(500, 32, 250, 32, kernel_size=kernel_size),
            ConvBlock(250, 32, 50, 1, kernel_size=kernel_size),
            nn.Flatten(),
        )
        kernel_size = 3
        self.espec_conv = nn.Conv2d(1, 16, kernel_size, padding=get_same_padding(kernel_size), bias=False)
        self.pspec_conv = nn.Conv2d(1, 16, kernel_size, padding=get_same_padding(kernel_size), bias=False)
        self.spec_net = nn.Sequential(
            ConvBlock2d(100, 16, 50, 32, kernel_size=kernel_size),
            ConvBlock2d(50, 32, 10, 32, kernel_size=kernel_size),
            ConvBlock2d(10, 32, 2, 16, kernel_size=kernel_size),
            nn.Flatten(),
            nn.Linear(4*16, 4*8),
            nn.ReLU(),
            nn.Linear(4*8, 25),
            nn.ReLU(),
        )
        self.est_conv = nn.Conv2d(1, 16, kernel_size, padding=get_same_padding(kernel_size), bias=False)
        self.pmfccs_conv = nn.Conv2d(1, 16, kernel_size, padding=get_same_padding(kernel_size), bias=False)
        self.mix_net = nn.Sequential(
            ConvBlock2d(100, 16, 50, 32, kernel_size=kernel_size),
            ConvBlock2d(50, 32, 10, 32, kernel_size=kernel_size),
            ConvBlock2d(10, 32, 2, 16, kernel_size=kernel_size),
            nn.Flatten(),
            nn.Linear(8*16, 4*8),
            nn.ReLU(),
            nn.Linear(4*8, 25),
            nn.ReLU(),
        )
        self.clf = nn.Sequential(
            nn.Flatten(),
            Linear(100, n_classes),
            nn.Softmax(dim=1)
        )
        self.clf_loss = nn.MSELoss()

    def forward(self, ecg, pcg, ecg_spec, pcg_spec, ecg_st, pcg_mfccs, y_true):
        ecg = self.ecg_conv(ecg)
        pcg = self.pcg_conv(pcg[:,:,::4])
        raw = torch.cat([ecg, pcg], dim=2)
        ecg_spec = self.espec_conv(ecg_spec)
        pcg_spec = self.pspec_conv(pcg_spec)
        spec = torch.cat([ecg_spec, pcg_spec], dim=3)
        pcg_mfccs = self.pmfccs_conv(pcg_mfccs)
        ecg_st = self.est_conv(ecg_st)
        mix = torch.cat([pcg_mfccs, ecg_st], dim=3)
        raw_fea = self.raw_net(raw)
        spec_fea = self.spec_net(spec)
        mix_fea = self.mix_net(mix)
        fea = torch.cat([raw_fea, spec_fea, mix_fea], dim=1)
        y_pred = self.clf(fea)
        loss = self.clf_loss(y_true, y_pred)
        return y_pred, loss, torch.zeros(1)

class GAP(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = x.mean(1)
        return x

class BNNECG(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        conv_layer_num = 5
        filter_width = 11
        keep = 0.5
        convs = []
        in_channels = 1
        fea_in = 2000
        # layer_norms = [[64, 1990],[64, 985],[110, 482],[110, 231],[156, 105],[156, 42],[202, 11]]
        for l in range(conv_layer_num):
            cnn_name = 'cnn_'+str(l)
            p_name = 'avgpool_' + str(l)
            drop_name = 'dropout_' + str(l)
            # bn_name = 'bnorm_' + str(l)
            ln_name = 'lnorm_' + str(l)
            if l == 0:
                out_channels = 64
            elif l % 2 == 0:
                out_channels = in_channels + 46
            else:
                out_channels = in_channels
            convs.append((
                cnn_name, 
                nn.Conv1d(in_channels, out_channels, filter_width)
            ))
            fea_in = conv_size(fea_in, filter_width, stride=1, padding=0)
            convs.append((
                ln_name,
                nn.LayerNorm([out_channels, int(fea_in)]),
            ))
            convs.append((
                p_name,
                nn.AvgPool1d(2, 2)
            ))
            fea_in = fea_in//2
            if l != conv_layer_num-1:
                convs.append((
                    drop_name,
                    nn.Dropout(keep),
                ))
            in_channels = out_channels
        self.convs = nn.Sequential(OrderedDict(convs))
        self.output = nn.Sequential(
            GAP(),
            nn.Dropout(keep),
            nn.Linear(int(fea_in),2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.output(x)
        return x
        


mod_list = {
    "RTNet": RTNet,
    "ECGResCLF": ECGResCLF,
    "PCGResCLF": PCGResCLF,
    "HanLi": HanLiNet,
    "BNNECG": BNNECG,
}
