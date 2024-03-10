import pathlib
import logging

import numpy as np

import s3prl.hub
import torch
import torch.nn.functional as F
import torchaudio

class WavEMADataset(torch.utils.data.Dataset):
    def __init__(self, 
                 wav_root="/data/all_data/VCTK/", 
                 ema_root="/data/prabhune/VCTK/", 
                 ema_dir="mngu0_wlm_est",
                 feature_model="wavlm_large",
                 device=0):
        """
        Sets up dataset for wav2ema inversion.

        Expected directory structure:
        wav_root/
            {spkr_id}/
                wav_16/
                    file1.wav
                    file2.wav
                    ...
                ...
            ...
        ema_root/
            {spkr_id}/
                {ema_dir}/
                    file1.npy
                    file2.npy
                    ...
                ...
            ...
        """
        #logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
        self.device = device

        self.wav_root = pathlib.Path(wav_root)
        self.ema_root = pathlib.Path(ema_root)
        self.ema_dir = ema_dir

        print("Loading files from directories...")
        self.wav_files = sorted(self.get_files_ext(self.wav_root, "wav_16", "wav"))
        self.ema_files = sorted(self.get_files_ext(self.ema_root, self.ema_dir, "npy"))

        self.feature_model = getattr(s3prl.hub, feature_model)()
        self.feature_model = self.feature_model.model.feature_extractor.to(self.device)
    
    def __len__(self):
        return len(self.wav_files)
    
    @torch.no_grad()
    def __getitem__(self, idx):
        audio, _ = torchaudio.load(self.wav_files[idx])
        audio = audio.to(self.device)

        audio = torch.cat([audio[:, -16000:], audio], dim=1)

        feats = []
        for i in range(16000, audio.shape[1], 1800):
            curr_audio = audio[:, i - 16000:i+1800]
            if curr_audio.shape[1] < 1800:
                curr_audio = F.pad(curr_audio, (0, 1800 - curr_audio.shape[1]))
            feat = self.feature_model(curr_audio).to(self.device)
            #feat = feat.transpose(1, 2)
            feat = feat[:, :, -5:]
            feats.append(feat)
        audio_features = torch.cat(feats, dim=2)

        # audio_features = self.feature_model(audio).to(self.device)

        ema = torch.from_numpy(np.load(self.ema_files[idx])).to(self.device)
        ema = torch.transpose(ema, 0, 1).unsqueeze(0)

        return audio_features, ema

    def get_files_ext(self, root_dir, dir, ext):
        return [f for f in root_dir.rglob(f"*{dir}/*.{ext}")]
    
if __name__ == "__main__":
    wav_ema_dataset = WavEMADataset()
    print(len(wav_ema_dataset))
    print(wav_ema_dataset.wav_files[:5])
    print(wav_ema_dataset.ema_files[:5])