import os
import gc
from tqdm import tqdm
import pickle
import librosa

import torch
import torch.nn as nn
import numpy as np

import s3prl.hub as hub

import soundfile as sf
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from scipy.signal import butter, filtfilt
from scipy.stats.stats import pearsonr


def get_files_recur(files_path, ext):
    files = []
    for root, _, filenames in os.walk(files_path):
        files.extend([os.path.join(root, f) for f in filenames if f.endswith(ext)])
    return files

class EMADataset():

    def __init__(
            self, 
            model, 
            data_path, 
            ema_path="ema\\",
            save_features=False, 
            five_min_train=False,
            train_ratio=0.8,
            layer=9, 
            device=0):
        self.data_path = data_path
        self.device = device

        self.sr = 16000

        print("Getting wav files")
        self.wav_files = get_files_recur(self.data_path, '.wav')
        self.wav_names = [Path(f).stem for f in tqdm(self.wav_files)]
        print("Done")

        print("Getting EMA files")
        self.ema_files = get_files_recur(self.data_path, '.npy')
        self.ema_files = [f for f in self.ema_files if ema_path in f]
        self.ema_names = [Path(f).stem for f in self.ema_files if f.endswith('.npy') and ema_path in f]
        print("Done")
        
        self.valid_files = list(set(self.wav_names) & set(self.ema_names))

        if save_features:
            print("Saving speech features")
            self.save_wavlm_features(model, layer)

        feats = []
        emas = []
        trg_sr = 50
        ema_sr = 200
        ft_sr = 50

        for file in tqdm(self.ema_files):
            ema_cut_len = len(ema_path) - 1 if '\\' in ema_path else len(ema_path)
            feat = np.load(f"{os.path.dirname(file)[:-ema_cut_len]}/feat_hb/{Path(file).stem}.npy")
            ema = np.load(file)

            if trg_sr < ema_sr:
                target = EMADataset.downsample_by_mean(ema, int(ema_sr/trg_sr))
            else:
                target = ema

            if trg_sr < ft_sr:
                feature = EMADataset.downsample_by_mean(feat, int(ft_sr/trg_sr))
            elif trg_sr > ft_sr:
                feature = EMADataset.upsample(feat, int(trg_sr/ft_sr))
            else:
                feature = feat

            data_len = min(target.shape[0], feature.shape[0])
            target = target[:data_len]
            feature = EMADataset.butter_bandpass_filter(feature, 6, 50)
            feature = feature[:data_len]

            feats.append(feature)
            emas.append(target)
        
        self.feats = np.concatenate(feats, 0)
        self.emas = np.concatenate(emas, 0)

        print(f"Total feats shape: {self.feats.shape}")
        print(f"Total emas shape: {self.emas.shape}")

        train_len = int(self.emas.shape[0] * train_ratio)
        if five_min_train:
            train_len = trg_sr * 5 * 60
        self.feat_train, self.feat_test = self.feats[0:train_len], self.feats[train_len:]
        self.ema_train, self.ema_test = self.emas[0:train_len], self.emas[train_len:]

        print(f"Train Length: {self.feat_train.shape[0]}")
        print(f"Test Length: {self.feat_test.shape[0]}")
    
    def 
    
    def set_train_ratio(self, train_ratio):
        train_len = int(self.emas.shape[0] * train_ratio)
        self.feat_train, self.feat_test = self.feats[0:train_len], self.feats[train_len:]
        self.ema_train, self.ema_test = self.emas[0:train_len], self.emas[train_len:]

    def save_wavlm_features(self, model, layer):
        self.wavs = [librosa.load(f, sr=self.sr)[0] for f in tqdm(self.wav_files)]
        for i in tqdm(range(len(self.wavs))):
            if os.path.exists(f"{os.path.dirname(self.wav_files[i])[:-4]}/feat_hb/{self.wav_names[i]}.npy"):
                continue
            wav = torch.from_numpy(self.wavs[i]).to(self.device)
            sample_emb = model([wav])["hidden_states"][layer]
            sample_emb = sample_emb.squeeze(0)

            np.save(f"{os.path.dirname(self.wav_files[i])[:-4]}/feat_hb/{self.wav_names[i]}.npy", sample_emb.detach().cpu())

    def downsample_by_mean(arr, factor):
        # arr : T,d
        T, d = arr.shape
        arr = arr[:int(T//factor*factor)]
        arr = arr.reshape(T//factor,factor, d)
        arr = arr.mean(1)
        return arr

    def upsample(arr,factor):
        # arr: T,d
        T, d = arr.shape
        output = np.zeros((T*factor,d))
        arr = np.concatenate([arr, arr[-1:]],0)
        for f in range(factor):
            output[np.arange(T)*factor+f] = (1-f/factor)*arr[:-1] + (f/factor)*arr[1:]
        return output

    def butter_bandpass(cut, fs, order=5):

        if isinstance(cut,list) and len(cut) == 2:
            return butter(order, cut, fs=fs, btype='bandpass')
        else:
            return butter(order, cut, fs=fs, btype='low')

    def butter_bandpass_filter(data, cut, fs, order=5):
        b, a = EMADataset.butter_bandpass(cut, fs, order=order)
        y = filtfilt(b, a, data,axis=0)
        return y