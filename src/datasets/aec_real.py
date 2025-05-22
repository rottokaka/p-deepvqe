import torch
import torchaudio
import torch.nn as nn
import torchaudio.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import numpy as np

class AECRealDataset(Dataset):
    def __init__(self, config, mode):
        """
        metadata: A pandas DataFrame containing 7 columns, including:
        - GUID [0]
        - Path to enrollment audio [1]
        - Path to microphone near-end signal [2]
        - Paths to microphone and loopback signals of the far-end [3, 4]
        - Paths to microphone and loopback signals of the far-end with echo [5, 6]
        """
        super().__init__()
        self.config = config
        if mode == "train":
            self.metadata = pd.read_csv(config.metadata_train, delimiter="\t")
        elif mode == "val":
            self.metadata = pd.read_csv(config.metadata_val, delimiter="\t")
        elif mode == "test":
            self.metadata = pd.read_csv(config.metadata_train, delimiter="\t")
        elif mode == "infer":
            self.metadata = pd.read_csv(config.metadata_train, delimiter="\t")
        
        if config.feat.window_fn == "Hann":
            window_fn = torch.hann_window
        elif config.feat.window_fn == "Hamming":
            window_fn = torch.hamming_window

        self.transform = torchaudio.transforms.Spectrogram(
            n_fft=config.feat.n_fft,
            hop_length=config.feat.hop_length,
            win_length=config.feat.win_length,
            window_fn=window_fn,
            power=None,
        )

        self.specaugment = torchaudio.transforms.SpecAugment(
            n_time_masks=self.config.augment.n_time_masks,
            time_mask_param=self.config.augment.time_mask_param,
            n_freq_masks=self.config.augment.n_freq_masks,
            freq_mask_param=self.config.augment.freq_mask_param,
            p=self.config.augment.specaugment_rate,
        )

    def __len__(self):
        return len(self.metadata)*2 # echo and no echo
    
    def __getitem__(self, index):
        if index < len(self)//2:
            farend_mic_sig, farend_mic_sr = torchaudio.load(self.metadata.iloc[index, 3])
            farend_lpb_sig, farend_lpb_sr = torchaudio.load(self.metadata.iloc[index, 4])
        else:
            index -= len(self)//2
            farend_mic_sig, farend_mic_sr = torchaudio.load(self.metadata.iloc[index, 5])
            farend_lpb_sig, farend_lpb_sr = torchaudio.load(self.metadata.iloc[index, 6])
        enrl_sig, enrl_sr = torchaudio.load(self.metadata.iloc[index, 1])
        nearend_mic_sig, nearend_mic_sr = torchaudio.load(self.metadata.iloc[index, 2])
        assert enrl_sr == nearend_mic_sr == farend_mic_sr == farend_lpb_sr == self.config.feat.target_sample_rate

        mic_sig, target_sig = self._mix_signal(nearend_mic_sig, farend_mic_sig)

        diff = mic_sig.shape[1]-farend_lpb_sig.shape[1]
        if diff > 0:
            farend_lpb_sig = self._pad(farend_lpb_sig, diff, False)
        elif diff < 0:
            mic_sig = self._pad(mic_sig, -diff, False)

        rate = random.uniform(0, 1)
        if rate <= self.config.augment.white_noise_rate:
            mic_sig = self._add_white_noise(mic_sig)

        mic_spec = self._get_spec(mic_sig)
        enrl_spec = self._get_spec(enrl_sig)
        farend_lpb_spec = self._get_spec(farend_lpb_sig)
        target_spec = self._get_spec(target_sig)
        enrl_spec = self._apply_specaugment(enrl_spec)

        return {"data": [enrl_spec, mic_spec, farend_lpb_spec, target_spec],
                "length": [torch.tensor([enrl_spec.shape[2]]), torch.tensor([mic_spec.shape[2]]), 
                           torch.tensor([farend_lpb_spec.shape[2]]), torch.tensor([target_spec.shape[2]])]}
    
    def _get_spec(self, signal):
        spec = self.transform(signal)
        return torch.stack([spec.real, spec.imag], dim=-1).contiguous()
    
    def _pad(self, signal, padding_length, left_pad=True):
        pad = torch.zeros((signal.shape[0], padding_length))
        if left_pad:
            return torch.concatenate((pad, signal), dim=1)
        else:
            return torch.concatenate((signal, pad), dim=1)
        
    def _add_white_noise(self, signal):
        # signal_power = signal.pow(2).mean()
        # snr = random.randint(self.config.min_aug_snr, self.config.max_aug_snr)
        # snr_power = 10 ** (snr / 10)
        # noise_power = signal_power / snr_power
        # noise = torch.randn_like(signal) * torch.sqrt(noise_power)
        # return signal + noise
        snr = random.randint(self.config.augment.min_aug_snr, self.config.augment.max_aug_snr)
        noise = torch.rand_like(signal) - 0.5
        audio_rms = signal.norm(p=2)
        noise_rms = noise.norm(p=2)
        
        snr = 10 ** (snr / 10)
        scale = audio_rms / (noise_rms * snr)
        noisy_audio = signal + scale * noise
        return noisy_audio
            
    def _mix_signal(self, nearend, farend):
        snr = torch.tensor([random.randint(self.config.augment.min_snr, self.config.augment.max_snr)])
        if nearend.shape[1] > farend.shape[1]:
            farend = self._pad(farend, nearend.shape[1]-farend.shape[1], False)
            mic = F.add_noise(nearend, farend, snr)
        else:
            start_sample = random.randint(0, farend.shape[1]-nearend.shape[1])
            nearend = self._pad(nearend, farend.shape[1]-nearend.shape[1]-start_sample, False)
            nearend = self._pad(nearend, start_sample)
            mic = F.add_noise(nearend, farend, snr)
        return mic, nearend
    
    def _apply_specaugment(self, spec):
        mask = torch.ones_like(spec[..., 0])
        mask = self.specaugment(mask)
        spec[..., 0] *= mask
        spec[..., 0] *= mask
        return spec

class AECRealModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        if self.config.metadata_train is not None:
            self.train_dataset = AECRealDataset(self.config, "train")
        if self.config.metadata_val is not None:
            self.val_dataset = AECRealDataset(self.config, "val")
        if self.config.metadata_test is not None:
            self.test_dataset = AECRealDataset(self.config, "test")
        if self.config.metadata_infer is not None:
            self.predict_dataset = AECRealDataset(self.config, "infer")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            batch_size=self.config.batch_size_per_gpu,
            num_workers=self.config.num_workers,
            shuffle=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            batch_size=self.config.batch_size_per_gpu,
            num_workers=self.config.num_workers,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            batch_size=self.config.batch_size_per_gpu,
            num_workers=self.config.num_workers,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            batch_size=self.config.batch_size_per_gpu,
            num_workers=self.config.num_workers,
        )

def build_data_module(config):
    return AECRealModule(config)

def collate_fn(batch):
    all_data = [[], [], [], []] # enrl, mic, farend_lpb, target
    all_length = [[], [], [], []]
    for item in batch:
        for idx in range(len(item["data"])):
            all_data[idx].append(item["data"][idx])
        for idx in range(len(item["length"])):
            all_length[idx].append(item["length"][idx])

    for idx in range(len(all_data)):
        max_length = int(max(all_length[idx]))
        all_data[idx] = torch.concatenate([
            nn.functional.pad(t, (0, 0, max_length-t.shape[2], 0)) # t.shape = [B, F, T, 2]
            for t in all_data[idx]
        ], dim=0).contiguous()
        all_length[idx] = torch.stack(all_length[idx], dim=0).contiguous()

    batch = {
        "data": all_data,
        "length": all_length,
    }

    return batch

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)