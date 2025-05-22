# ------------------------------------------------------------------------
# Modified from DeepVQE (https://github.com/Xiaobin-Rong/deepvqe.git)
# Copyright (c) 2025 Rong Xiaobin. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from einops import rearrange


class FE(nn.Module):
    """Feature extraction"""
    def __init__(self, c=0.3):
        super().__init__()
        self.c = c

    def forward(self, x):
        """x: (B,F,T,2)"""
        x_mag = torch.sqrt(x[...,[0]]**2 + x[...,[1]]**2 + 1e-12)
        x_c = torch.div(x, x_mag.pow(1-self.c) + 1e-12)
        return x_c.permute(0,3,2,1).contiguous()


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pad = nn.ZeroPad2d([1,1,3,0])
        self.conv = nn.Conv2d(channels, channels, kernel_size=(4,3))
        self.bn = nn.BatchNorm2d(channels)
        self.elu = nn.ELU()

    def forward(self, x):
        """x: (B,C,T,F)"""
        y = self.elu(self.bn(self.conv(self.pad(x))))
        return y + x
    

class InvertedResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels//2),
            nn.ZeroPad2d([1, 1, 3, 0]),
            nn.Conv2d(channels//2, channels//2, kernel_size=(4,3)),
            nn.BatchNorm2d(channels//2),
            nn.Hardswish(),
            nn.Conv2d(channels//2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels))

    def forward(self, x):
        """x: (B,C,T,F)"""
        return x + self.block(x)
        

class AlignBlock(nn.Module):
    def __init__(self, in_channels_mic, in_channels_ref, hidden_channels, delay=100):
        super().__init__()
        self.pconv_mic = nn.Conv2d(in_channels_mic, hidden_channels, 1)
        self.pconv_ref = nn.Conv2d(in_channels_ref, hidden_channels, 1)
        self.pconv_val = nn.Conv2d(in_channels_ref, hidden_channels, 1)
        self.unfold = nn.Sequential(nn.ZeroPad2d([0,0,delay-1,0]),
                                    nn.Unfold((delay, 1)))
        self.conv = nn.Sequential(nn.ZeroPad2d([1,1,4,0]),
                                  nn.Conv2d(hidden_channels, 1, (5,3)))
        
    def forward(self, x_mic, x_ref):
        """
        x_mic: (B,C,T,F)
        x_ref: (B,C,T,F)
        """
        Q = self.pconv_mic(x_mic)  # (B,H,T,F)
        K = self.pconv_ref(x_ref)  # (B,H,T,F)
        Ku = self.unfold(K)        # (B, H*D, T*F)
        Ku = Ku.view(K.shape[0], K.shape[1], -1, K.shape[2], K.shape[3])\
            .permute(0,1,3,2,4).contiguous()  # (B,H,T,D,F)
        V = torch.sum(Q.unsqueeze(-2) * Ku, dim=-1)      # (B,H,T,D)
        V = self.conv(V)           # (B,1,T,D)
        A = torch.softmax(V, dim=-1)[..., None]  # (B,1,T,D,1)
        
        Val = self.pconv_val(x_ref)
        y = self.unfold(Val).view(K.shape[0], K.shape[1], -1, K.shape[2], K.shape[3])\
                .permute(0,1,3,2,4).contiguous()  # (B,H,T,D,F)
        y = torch.sum(y * A, dim=-2)
        return y


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4,3), stride=(1,2)):
        super().__init__()
        self.pad = nn.ZeroPad2d([1,1,3,0])
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        self.invresblock = InvertedResidualBlock(out_channels)

    def forward(self, x):
        return self.invresblock(self.elu(self.bn(self.conv(self.pad(x)))))


class Bottleneck(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True)
        self.ln1 = nn.LayerNorm(input_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(input_size+hidden_size, input_size)
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, x, e = None):
        """x : (B,C,T,F)"""
        y = rearrange(x, 'b c t f -> b t (c f)')
        y = self.ln1(y)
        is_first_pass = False
        if e == None:
            is_first_pass = True
            e = torch.zeros(list(y.shape[:-1])+[self.hidden_size])
        y = torch.cat([y, e.expand(-1, y.shape[1], -1)], dim=-1)
        y = self.fc1(y)
        y = self.gru(y)[0]
        y = self.ln2(y)

        if is_first_pass:
            # internal embedding
            e = torch.mean(y, dim=1, keepdim=True)
            return e
        y = self.fc2(y)
        y = rearrange(y, 'b t (c f) -> b c t f', c=x.shape[1])
        return y
    

class SubpixelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4,3)):
        super().__init__()
        self.pad = nn.ZeroPad2d([1,1,3,0])
        self.conv = nn.Conv2d(in_channels, out_channels*2, kernel_size)
        
    def forward(self, x):
        y = self.conv(self.pad(x))
        y = rearrange(y, 'b (r c) t f -> b c t (r f)', r=2)
        return y
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cache_channels, kernel_size=(4,3), is_last=False):
        super().__init__()
        self.skip_conv = nn.Conv2d(cache_channels, in_channels, 1)
        self.invresblock = InvertedResidualBlock(in_channels)
        self.deconv = SubpixelConv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        self.is_last = is_last

    def forward(self, x, x_en):
        y = x + self.skip_conv(x_en)
        y = self.deconv(self.invresblock(y))
        if not self.is_last:
            y = self.elu(self.bn(y))
        return y
    

class CCM(nn.Module):
    """Complex convolving mask block"""
    def __init__(self):
        super().__init__()
        self.v = torch.tensor([1, -1/2 + 1j*np.sqrt(3)/2, -1/2 - 1j*np.sqrt(3)/2])
        self.unfold = nn.Sequential(nn.ZeroPad2d([1,1,2,0]),
                                    nn.Unfold(kernel_size=(3,3)))
    
    def forward(self, m, x):
        """
        m: (B,27,T,F)
        x: (B,F,T,2)"""
        m = rearrange(m, 'b (r c) t f -> b r c t f', r=3)
        H = torch.sum(self.v.to(m.device)[None,:,None,None,None] * m, dim=1)  # (B,C/3,T,F), complex
        M = rearrange(H, 'b (m n) t f -> b m n t f', m=3)  # (B,m,n,T,F), complex
        
        x = x.permute(0,3,2,1).contiguous()  # (B,2,T,F), real
        x = torch.complex(x[:,0], x[:,1])    # (B,T,F), complex
        x_unfold = self.unfold(x[:,None])
        x_unfold = rearrange(x_unfold, 'b (m n) (t f) -> b m n t f', m=3,f=x.shape[-1])
        
        x_enh = torch.sum(M * x_unfold, dim=(1,2))  # (B,T,F), complex
        x_enh = torch.stack([x_enh.real, x_enh.imag], dim=3).transpose(1,2).contiguous()
        return x_enh


class PersonalizedDeepVQE(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.fe = FE(config.c)
        assert len(config.enc.mic)+len(config.enc.mix) == len(config.dec)

        # mic signal
        self.enblocks_mic = []
        for i in range(len(config.enc.mic)):
            if i == 0:
                self.enblocks_mic.append(EncoderBlock(2, config.enc.mic[i]))
            else:
                self.enblocks_mic.append(EncoderBlock(config.enc.mic[i-1], config.enc.mic[i]))

        # far end signal
        self.enblocks_ref = []
        for i in range(len(config.enc.ref)):
            if i == 0:
                self.enblocks_ref.append(EncoderBlock(2, config.enc.ref[i]))
            else:
                self.enblocks_ref.append(EncoderBlock(config.enc.ref[i-1], config.enc.ref[i]))

        # aligned far end signal
        self.alignblock = AlignBlock(config.enc.mic[-1], config.enc.ref[-1], config.enc.mic[-1], config.enc.delay)

        # combined signal
        self.enblocks_mix = []
        for i in range(len(config.enc.mix)):
            if i == 0:
                self.enblocks_mix.append(EncoderBlock(config.enc.mic[-1]*2, config.enc.mix[i]))
            else:
                self.enblocks_mix.append(EncoderBlock(config.enc.mix[i-1], config.enc.mix[i]))
        
        freq_dim = self._get_freq_dim(config.enc.n_fft, len(self.enblocks_mic)+len(self.enblocks_mix))
        self.bottle = Bottleneck(config.enc.mix[-1]*freq_dim, config.bottle_neck.dim)
        
        # decoder
        en_channels_reversed = list(config.enc.mic+config.enc.mix)[::-1]
        self.deblocks = []
        for i in range(len(config.dec)):
            if i == 0:
                self.deblocks.append(DecoderBlock(config.en.mix[-1], config.dec[i], en_channels_reversed[i]))
            elif i == len(config.dec) - 1:
                self.deblocks.append(DecoderBlock(config.dec[i-1], config.dec[i], en_channels_reversed[i], is_last=True))
            else:
                self.deblocks.append(DecoderBlock(config.dec[i-1], config.dec[i], en_channels_reversed[i]))

        # ccm
        self.ccm = CCM()

    def _get_freq_dim(self, n_fft, times):
        freq_dim = n_fft // 2 + 1
        for i in range(times):
            freq_dim = (freq_dim - 3 + 2) // 2 +1
        return freq_dim
        
    def forward(self, x_enr, x_mic_raw, x_ref=None):
        """x: (B,F,T,2)"""

        x_enr = self.fe(x_enr)
        x_mic = self.fe(x_mic)
        if x_ref is None:
            x_ref = torch.zeros_like(x_mic)
        else:
            x_ref = self.fe(x_ref)
        tmp = torch.zeros_like(x_enr)

        # enrollment signal path
        for block in self.enblocks_mic:
            x_enr = block(x_enr)

        for block in self.enblocks_ref:
            tmp = block(tmp)

        tmp = self.alignblock(x_enr, tmp)
        x_enr = torch.concatenate([x_enr, tmp], dim=1).contiguous()

        for block in self.enblocks_mix:
            x_enr = block(x_enr)

        emb = self.bottle(x_enr)

        # mic signal path
        enc_cache = [x_mic]
        for block in self.enblocks_mic:
            enc_cache.append(block(enc_cache[-1]))

        # ref signal path
        for block in self.enblocks_ref:
            x_ref = block(x_ref)

        x_ref = self.alignblock(enc_cache[-1], x_ref)

        x_mix = torch.concatenate([enc_cache[-1], x_ref], dim=1).contiguous()

        for block in self.enblocks_mix:
            x_mix = block(x_mix)
            enc_cache.append(x_mix)

        x_mix = self.bottle(x_mix, emb)
        enc_cache = enc_cache[::-1]

        # decoder
        for idx in range(len(self.deblocks)):
            x_mix = self.deblocks[idx](x_mix, enc_cache[idx])[..., :enc_cache[idx+1].shape[-1]]

        x_enh = self.ccm(x_mix, x_mic_raw) # (B,F,T,2)
        
        return x_enh
    
    def training_step(self, batch, batch_idx):
        return

def build_model(config):
    return PersonalizedDeepVQE(config)

def input_constructor(input_res):
    # input_res is just a placeholder, e.g., (3, 224, 224)
    x1 = torch.randn(1, 257, 100, 2)
    x2 = torch.randn(1, 257, 100, 2)  # e.g., an auxiliary vector input
    x3 = torch.randn(1, 257, 80, 2)
    return dict(x_mic=x1, x_ref=x2, x_enr=x3)  # or return (x1, x2) depending on model

if __name__ == "__main__":
    model = PersonalizedDeepVQE().eval()
    x = torch.randn(1, 257, 63, 2)
    y = model(x, x)

    
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (257, 63, 2), as_strings=True,
                                           input_constructor=input_constructor,
                                           print_per_layer_stat=False, verbose=True)
    print(flops, params)

    """causality check"""
    a = torch.randn(1, 257, 100, 2)
    b = torch.randn(1, 257, 100, 2)
    c = torch.randn(1, 257, 100, 2)
    x1 = torch.cat([a, b], dim=2)
    x2 = torch.cat([a, c], dim=2)
    y1 = model(x1, x2)
    y2 = model(x2, x1)
    print((y1[:,:,:100,:] - y2[:,:,:100,:]).abs().max())
    print((y1[:,:,100:,:] - y2[:,:,100:,:]).abs().max())
