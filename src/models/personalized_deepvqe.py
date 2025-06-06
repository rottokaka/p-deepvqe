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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchaudio

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
            nn.Conv2d(channels, channels*2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels*2),
            nn.ZeroPad2d([1, 1, 3, 0]),
            nn.Conv2d(channels*2, channels*2, kernel_size=(4,3)),
            nn.BatchNorm2d(channels*2),
            nn.ELU(),
            nn.Conv2d(channels*2, channels, kernel_size=1, bias=False),
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

    def forward(self, x, lengths, e=None):
        """x : (B,C,T,F)"""
        y = rearrange(x, 'b c t f -> b t (c f)')
        y = self.ln1(y)
        is_first_pass = False
        if e == None:
            is_first_pass = True
            e = torch.zeros(list(y.shape[:-1])+[self.hidden_size])
            e = e.to(y.device)
        y = torch.cat([y, e.expand(-1, y.shape[1], -1)], dim=-1)
        y = self.fc1(y)
        y = pack_padded_sequence(y, lengths.squeeze(1), batch_first=True, enforce_sorted=False)
        y = self.gru(y)[0]
        y, _ = pad_packed_sequence(y, batch_first=True)
        y = self.ln2(y)

        if is_first_pass:
            # internal embedding
            e = torch.zeros((y.shape[0], y.shape[2]), dtype=torch.float, device=y.device)
            for i in range(e.shape[0]):
                e[i] = torch.mean(y[0, :lengths[i]], dim=0)
            # e = torch.mean(y, dim=1, keepdim=True)
            return e.unsqueeze(1)
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
        x: (B,F,T,2)
        """
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
        self.loss_fn = ComplexCompressedMSELoss(config.loss_func)
        config = config.arch
        self.fe = FE(config.c)
        assert len(config.enc.mic)+len(config.enc.mix) == len(config.dec)

        # mic signal
        self.enblocks_mic = nn.ModuleList()
        for i in range(len(config.enc.mic)):
            if i == 0:
                self.enblocks_mic.append(EncoderBlock(2, config.enc.mic[i]))
            else:
                self.enblocks_mic.append(EncoderBlock(config.enc.mic[i-1], config.enc.mic[i]))

        # far end signal
        self.enblocks_ref = nn.ModuleList()
        for i in range(len(config.enc.ref)):
            if i == 0:
                self.enblocks_ref.append(EncoderBlock(2, config.enc.ref[i]))
            else:
                self.enblocks_ref.append(EncoderBlock(config.enc.ref[i-1], config.enc.ref[i]))

        # aligned far end signal
        self.alignblock = AlignBlock(config.enc.mic[-1], config.enc.ref[-1], config.enc.mic[-1], config.enc.delay)

        # combined signal
        self.enblocks_mix = nn.ModuleList()
        for i in range(len(config.enc.mix)):
            if i == 0:
                self.enblocks_mix.append(EncoderBlock(config.enc.mic[-1]*2, config.enc.mix[i]))
            else:
                self.enblocks_mix.append(EncoderBlock(config.enc.mix[i-1], config.enc.mix[i]))
        
        freq_dim = self._get_freq_dim(config.enc.n_fft, len(self.enblocks_mic)+len(self.enblocks_mix))
        self.bottle = Bottleneck(config.enc.mix[-1]*freq_dim, config.bottle_neck.dim)
        
        # decoder
        en_channels_reversed = list(config.enc.mic+config.enc.mix)[::-1]
        self.deblocks = nn.ModuleList()
        for i in range(len(config.dec)):
            if i == 0:
                self.deblocks.append(DecoderBlock(config.enc.mix[-1], config.dec[i], en_channels_reversed[i]))
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
        
    def forward(self, x_enr, x_enr_length, x_mic_raw, x_mic_raw_length, x_ref=None):
        """x: (B,F,T,2)"""
        x_enr = self.fe(x_enr)
        x_mic = self.fe(x_mic_raw)
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
        emb = self.bottle(x_enr, x_enr_length)

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
        x_mix = self.bottle(x_mix, x_mic_raw_length, emb)
        enc_cache = enc_cache[::-1]

        # decoder
        for idx in range(len(self.deblocks)):
            x_mix = self.deblocks[idx](x_mix, enc_cache[idx])[..., :enc_cache[idx+1].shape[-1]]

        x_enh = self.ccm(x_mix, x_mic_raw) # (B,F,T,2)
        return x_enh
    
    def training_step(self, batch, batch_idx):
        enrl, mic, farend_lpb, target = batch["data"]
        enrl_length, mic_length, farend_lpb_length, target_length = batch["length"]
        pred = self(enrl, enrl_length, mic, mic_length, farend_lpb)
        # mask = torch.arange(mic.shape[2])[None, :].to(mic.device) < mic_length[:, None]
        # mask = mask[..., None]
        # mask.to(enrl.device)
        # pred = pred * mask
        loss = self.loss_fn(pred, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.current_epoch % 100 == 0:
            mic_sig = self._get_waveform(mic.clone().detach())
            farend_lpb_sig = self._get_waveform(farend_lpb.clone().detach())
            target_sig = self._get_waveform(target.clone().detach())
            pred_sig = self._get_waveform(pred.clone().detach())
            torchaudio.save(f"D:/RTV/PersonalizedDeepVQE/outputs/{self.current_epoch}_{batch_idx}_mic_{loss}.wav", mic_sig, 16000, bits_per_sample=16)
            torchaudio.save(f"D:/RTV/PersonalizedDeepVQE/outputs/{self.current_epoch}_{batch_idx}_ref_{loss}.wav", farend_lpb_sig, 16000, bits_per_sample=16)
            torchaudio.save(f"D:/RTV/PersonalizedDeepVQE/outputs/{self.current_epoch}_{batch_idx}_target_{loss}.wav", target_sig, 16000, bits_per_sample=16)
            torchaudio.save(f"D:/RTV/PersonalizedDeepVQE/outputs/{self.current_epoch}_{batch_idx}_pred_{loss}.wav", pred_sig, 16000, bits_per_sample=16)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=6e-4, weight_decay=1e-7)
    
    def _get_waveform(self, spec):
        transform = torchaudio.transforms.InverseSpectrogram(
            n_fft=480,
            hop_length=160,
            win_length=320,
            window_fn=torch.hann_window,
        )
        spec = torch.complex(spec[..., 0], spec[..., 1]).contiguous().cpu()
        # print(spec.shape)
        return transform(spec)
    
class ComplexCompressedMSELoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.compress = FE(config.c)
        self.beta = config.beta

    def _magnitude_missmatch(self, pred, target):
        pred_mag = torch.sqrt(pred[..., 0]**2+pred[..., 1]**2)
        target_mag = torch.sqrt(target[..., 0]**2+target[..., 1]**2)
        return torch.sum((pred_mag-target_mag)**2)

    def _phase_missmatch(self, pred, target):
        return torch.sum((pred[..., 0]-target[..., 0])**2+(pred[..., 1]-target[..., 1])**2)

    def forward(self, pred, target):
        """pred: (B,F,T,2)"""
        # pred = self.compress(pred)
        # target = self.compress(target)
        # loss = self._magnitude_missmatch(pred, target)+self.beta*self._phase_missmatch(pred, target)
        # return loss/pred.shape[0]/pred.shape[2]
        pred = torch.complex(pred[..., 0], pred[..., 1]).contiguous()
        target = torch.complex(target[..., 0], target[..., 1]).contiguous()
        return loss_function(pred, target, self.beta)

def power_law_compress(S, power=0.3, epsilon=1e-8):
    """
    Apply power-law compression to the STFT magnitude while preserving phase.
    
    Args:
        S (torch.Tensor): Complex STFT tensor of shape [batch, time, freq]
        power (float): Power for compression, default is 0.3
        epsilon (float): Small value to avoid division by zero
    
    Returns:
        S_pow (torch.Tensor): Compressed complex STFT
        mag_pow (torch.Tensor): Compressed magnitude
    """
    mag = torch.abs(S)
    mag_safe = mag + epsilon
    mag_pow = mag ** power
    S_pow = (mag_pow / mag_safe) * S
    return S_pow, mag_pow

def loss_function(S_enhanced, S_clean, lambda_val=0.113):
    """
    Compute the loss function between enhanced and clean STFTs.
    
    Args:
        S_enhanced (torch.Tensor): Enhanced STFT, shape [batch, time, freq], complex
        S_clean (torch.Tensor): Clean STFT, shape [batch, time, freq], complex
        lambda_val (float): Weight for the complex difference term, default is 0.113
    
    Returns:
        torch.Tensor: Loss value, averaged over the batch
    """
    # Compress both enhanced and clean STFTs
    S_enhanced_pow, mag_enhanced_pow = power_law_compress(S_enhanced)
    S_clean_pow, mag_clean_pow = power_law_compress(S_clean)
    
    # Magnitude difference loss
    mag_diff = mag_enhanced_pow - mag_clean_pow
    mag_loss = torch.sum(mag_diff ** 2, dim=[1, 2])  # Sum over time and freq
    
    # Complex difference loss
    complex_diff = S_enhanced_pow - S_clean_pow
    complex_loss = torch.sum(complex_diff.real ** 2 + complex_diff.imag ** 2, dim=[1, 2])
    
    # Total loss, averaged over batch
    total_loss = mag_loss + lambda_val * complex_loss
    return total_loss.mean()

def build_model(config):
    return PersonalizedDeepVQE(config)