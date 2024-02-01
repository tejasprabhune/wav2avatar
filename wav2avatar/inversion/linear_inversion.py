import os
import gc
import pickle
import pathlib
import librosa
from typing import overload
from tqdm import tqdm

import scipy
import torch
import torchaudio
import torch.nn as nn
import numpy as np

import s3prl.hub

import soundfile as sf

import matplotlib.pyplot as plt

import sklearn.linear_model as sklm


class EMADataset:

    """
    Constructs an EMA dataset from a given file structure of:

    /{data_root}
        /ema
            /*.npy
        /wav
            /*.npy
        /{ssl_feat}
            /*.npy

    Args:
        data_root: Root directory of dataset
        ssl_feat: SSL model name to use and for subdirectory name
        ema_sr: Original sample rate for EMA
        ssl_sr: Original sample rate for SSL features
        model: SSL model to use in generating features if needed
        layer: Layer of SSL model to use in generating features if needed
        device: CUDA device
        low_pass: Amount in Hz to use for low pass filter
        train_ratio: Amount of total data to use in training (rest in test)
    """

    def __init__(
        self,
        data_root: str,
        ssl_feat: str,
        ema_sr: int = None,
        ssl_sr: int = None,
        model=None,
        layer: int = 9,
        device: int = 0,
        low_pass: int = 6,
        train_ratio: float = 0.8,
    ):
        """
        Creates a train/test split for a new EMADataset. If SSL features are
        not present/complete, generates all SSL features.
        """
        torch.set_grad_enabled(False)
        self.sr = 16000
        self.device = device
        self.layer = layer
        self.ssl_feat = ssl_feat
        self.low_pass = low_pass

        self.model = model
        if self.model == None:
            self.model = self.get_model()

        self.root = pathlib.Path(data_root)
        self.ema_root = self.root / "ema"
        self.wav_root = self.root / "wav"
        self.ssl_root = self.root / ssl_feat
        self.ssl_root.mkdir(parents=True, exist_ok=True)

        get_stems = lambda root_dir: [f.stem for f in root_dir.rglob("*")]
        wav_stems = get_stems(self.wav_root)
        ema_stems = get_stems(self.ema_root)
        ssl_stems = get_stems(self.ssl_root)
        self.stems = list(set(wav_stems) & set(ema_stems))
        assert len(self.stems) >= 1

        if set(self.stems) >= set(ssl_stems):
            print(
                "SSL feature directory does not match wav/ema directories.",
                "Saving features..."
            )
            self.save_features()

        self.ema_sr = self.get_npy_sr(self.ema_root, ema_sr)
        self.ssl_sr = self.get_npy_sr(self.ssl_root, ssl_sr)
        self.trg_sr = min(self.ema_sr, self.ssl_sr)

        self.emas, self.feats = self.compile_ema_feat()

        print(f"Total feats shape: {self.feats.shape}")
        print(f"Total emas shape: {self.emas.shape}")

        train_len = int(self.emas.shape[0] * train_ratio)
        self.feat_train, self.feat_test = (
            self.feats[0:train_len],
            self.feats[train_len:],
        )
        self.ema_train, self.ema_test = (
            self.emas[0:train_len],
            self.emas[train_len:],
        )

        print(f"Train Length: {self.feat_train.shape[0]}")
        print(f"Test Length: {self.feat_test.shape[0]}")

    def compile_ema_feat(self):
        """Gets all ema and SSL feats with a bandpass filter on features."""
        emas = []
        feats = []

        print("Compiling all EMA and SSL features...")
        for stem in tqdm(self.stems):
            ema = np.load(self.ema_root / f"{stem}.npy")
            feat = np.load(self.ssl_root / f"{stem}.npy")

            ema, feat = self.match_ema_feat(ema, feat)
            min_len = min(len(ema), len(feat))
            ema = ema[:min_len]
            feat = EMADataset.butter_bandpass_filter(
                feat, self.low_pass, self.ssl_sr
            )
            feat = feat[:min_len]

            emas.append(ema)
            feats.append(feat)

        print("Concatenating EMA and SSL features...")
        emas = np.concatenate(emas, 0)
        feats = np.concatenate(feats, 0)
        return emas, feats

    def save_features(self):
        """
        Saves SSL feature .npys for each wav/ema pair in
        dataset (/{data_root}/{ssl_feat}/*.npy).
        """
        for stem in tqdm(self.stems):
            audio = EMADataset.load_audio(
                self.wav_root / f"{stem}.wav", self.sr, self.device
            )
            sample_emb = EMADataset.get_feature(audio, self.model, self.layer)
            np.save(self.ssl_root / f"{stem}.npy", sample_emb.cpu())

    def get_feature(audio: torch.Tensor, model, layer):
        """Get SSL feature given an audio."""
        return model([audio])["hidden_states"][layer].squeeze(0)

    def get_model(self):
        """Get model from s3prl."""
        return getattr(s3prl.hub, self.ssl_feat)().to(self.device)

    def load_audio(wav, sr, device):
        """Load audio tensor and resample to appropriate sample rate."""

        audio, sr = torchaudio.load(wav)
        audio = audio.to(device).squeeze(0)
        if sr != sr:
            audio = torchaudio.functional.resample(
                audio, orig_freq=sr, new_freq=sr
            )
        return audio

    def get_npy_sr(self, npy_dir: pathlib.Path, npy_sr: int):
        """
        Approximates dataset's sample rate for a given feature if sample
        rate is not given.

        Args:
            npy_dir (Path): Path to data for sr approximation
            npy_sr (int): Given sr to check if approximation should be done

        Returns:
            sr (int): Approximated sample rate of data
        """
        if npy_sr is not None:
            return npy_sr

        srs = []
        for i in range(10):
            first_stem = self.stems[i]
            audio = EMADataset.load_audio(
                self.wav_root / f"{first_stem}.wav", self.sr, self.device
            ).cpu()
            audio_len = len(audio) / self.sr

            npy = np.load(npy_dir / f"{first_stem}.npy")

        srs.append(len(npy) / audio_len)

        mean_sr = int(np.mean(srs))
        return round(mean_sr, -1) if mean_sr <= 100 else round(mean_sr, -2)

    def match_ema_feat(self, ema, feat):
        """Downsample/upsample EMA and SSL features to match SRs."""
        if self.trg_sr < self.ema_sr:
            ema = EMADataset.downsample_by_mean(
                ema, int(self.ema_sr / self.trg_sr)
            )

        if self.trg_sr < self.ssl_sr:
            feat = EMADataset.downsample_by_mean(
                feat, int(self.ssl_sr / self.trg_sr)
            )
        elif self.trg_sr > self.ssl_sr:
            feat = EMADataset.upsample(feat, int(self.trg_sr / self.ssl_sr))

        return ema, feat

    def downsample_by_mean(arr, factor):
        # arr : T,d
        T, d = arr.shape
        arr = arr[: int(T // factor * factor)]
        arr = arr.reshape(T // factor, factor, d)
        arr = arr.mean(1)
        return arr

    def upsample(arr, factor):
        # arr: T,d
        T, d = arr.shape
        output = np.zeros((T * factor, d))
        arr = np.concatenate([arr, arr[-1:]], 0)
        for f in range(factor):
            output[np.arange(T) * factor + f] = (1 - f / factor) * arr[:-1] + (
                f / factor
            ) * arr[1:]
        return output

    def butter_bandpass(cut, fs, order=5):
        if isinstance(cut, list) and len(cut) == 2:
            return scipy.signal.butter(order, cut, fs=fs, btype="bandpass")
        else:
            return scipy.signal.butter(order, cut, fs=fs, btype="low")

    def butter_bandpass_filter(data, cut, fs, order=5):
        b, a = EMADataset.butter_bandpass(cut, fs, order=order)
        y = scipy.signal.filtfilt(b, a, data, axis=0)
        return y


class LinearInversion:
    def __init__(
        self,
        ema_dataset=None,
        ssl_model="hubert",
        layer=9,
        ckpt="",
        sr=16000,
        device=0,
    ):
        torch.set_grad_enabled(False)
        self.ema_dataset = ema_dataset
        self.lr_model = sklm.LinearRegression()
        self.layer = layer
        self.sr = sr
        self.device = device

        if self.ema_dataset is not None and self.ema_dataset.model is not None:
            self.ssl_model = self.ema_dataset.model
        else:
            self.ssl_model = getattr(s3prl.hub, ssl_model)().to(self.device)

        self.ckpt = pathlib.Path(ckpt)
        if self.ckpt.is_file():
            with open(self.ckpt, "rb") as f:
                self.lr_model = pickle.load(f)
        else:
            print("ckpt not found!")

    def fit(self, val_report=False):
        self.lr_model.fit(
            self.ema_dataset.feat_train, self.ema_dataset.ema_train
        )

        if val_report:
            self.val_report()

    def val_report(self, avg=True):
        if len(self.ema_dataset.feat_test) == 0:
            return []
        ema_test_pred = self.lr_model.predict(self.ema_dataset.feat_test)

        corrs = []
        for feat_num in range(12):
            corrs.append(
                self.get_corr(
                    ema_test_pred, self.ema_dataset.ema_test, feat_num
                )[0]
            )

        if avg:
            return np.mean(corrs)
        return corrs

    def get_corr(self, prd, trg, feat_num):
        return scipy.stats.pearsonr(prd[:, feat_num], trg[:, feat_num])

    def mngu0_to_hprc(self, arr):
        for i in range(0, 12, 2):
            arr[:, i] *= -1

        arr_td = arr[:, 0:2]
        arr_tb = arr[:, 2:4]
        arr_tt = arr[:, 4:6]
        arr_li = arr[:, 6:8]
        arr_ul = arr[:, 8:10]
        arr_ll = arr[:, 10:12]

        arr[:, 0:2] = arr_li
        arr[:, 2:4] = arr_ul
        arr[:, 4:6] = arr_ll
        arr[:, 6:8] = arr_tt
        arr[:, 8:10] = arr_tb
        arr[:, 10:12] = arr_td

    def predict(self, wav: str):
        audio = EMADataset.load_audio(wav, self.sr, self.device)
        return EMADataset.butter_bandpass_filter(self.predict_from_tensor(audio), 10, 50)

    def predict_from_tensor(self, audio: torch.Tensor):
        feat = EMADataset.get_feature(audio, self.ssl_model, self.layer).cpu()
        return self.predict_from_feat(feat)
    
    def predict_from_file(self, npy_file):
        feat = np.load(npy_file)
        return self.predict_from_feat(feat)
    
    def predict_from_feat(self, feat):
        pred_ssl_ema = self.lr_model.predict(feat)
        self.mngu0_to_hprc(pred_ssl_ema)
        return EMADataset.butter_bandpass_filter(pred_ssl_ema, 10, 50)

    def save(self, file):
        with open(file, "wb") as f:
            pickle.dump(self.lr_model, f)


if __name__ == "__main__":
    #ema_dataset = EMADataset(
    #   "C:/Users/tejas/Documents/UCBerkeley/bci/mngu0",
    #   "wavlm_large",
    #   train_ratio=1.0,
    #)
    lr_model = LinearInversion(ckpt="ckpts/lr_hbb_l10_mng_all.pkl")
    #lr_model.fit()
    #lr_model.save("ckpts/lr_hbb_l10_mng_all.pkl")
    np.save("ema/mng_david_pred.npy", lr_model.predict("wav/david_audio.wav"))
