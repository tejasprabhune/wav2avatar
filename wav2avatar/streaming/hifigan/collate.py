import torch
import torch.utils.data
import numpy as np

from wav2avatar.streaming.hifigan.wav_ema_dataset import WavEMADataset

import typing


def collate(in_audio, context_len, step_size=5):
    """
    Goal: 
        Input: (1, 512, audio_len)
        Output: (audio_len // step_size, 512, context_len)
    """

    truncated_audio = in_audio[:, :, :-(in_audio.shape[2] % context_len) or None]
    #print(truncated_audio.shape)

    num_batches = in_audio.shape[2] // step_size
    collates = []
    for batch in range(num_batches):
        batch_start = batch * step_size
        if (batch_start + context_len) > truncated_audio.shape[2]:
            break
        collates.append(truncated_audio[:, :, batch_start:(batch_start + context_len)])
    collated = torch.concatenate(collates, dim=0)

    return collated

def uncollate(collated, step_size=50):
    pieces = [collated[0]]
    for i in range(1, len(collated)):
        piece = collated[i]
        pieces.append(piece[:, -step_size:])
    return torch.concatenate(pieces, dim=1).unsqueeze(0)

def new_car_collate(batch, step_size=5):
    audio, ema = zip(*batch)


    batch_audios = list(audio)
    batch_emas = list(ema)

    collated_audios = []
    collated_emas = []
    collated_ema_ars = []
    for i in range(len(batch_audios)):
        input_audio = batch_audios[i].detach().cpu()
        seq_len = input_audio.shape[2]
        input_ema = batch_emas[i].detach().cpu()

        input_ema_ar = torch.concatenate([torch.zeros(1, 12, step_size), input_ema[:, :, :seq_len-step_size]], dim=2)

        # audio = torch.cat(audio, dim=2)
        # ema = torch.cat(ema, dim=2)

        audios = []
        ema_ars = []
        emas = []


        truncated_seq_len = (seq_len - (seq_len % step_size)) - 50
        #print(seq_len, truncated_seq_len)
        for i in range(0, truncated_seq_len, step_size):
            curr_audio = input_audio[:, :, i:i+50 or None]
            audios.append(curr_audio)

            curr_ema_ar = input_ema_ar[:, :, i:i+50 or None]
            ema_ars.append(curr_ema_ar)

            curr_ema = input_ema[:, :, i+50-step_size:i+50 or None]
            emas.append(curr_ema)

        audios = torch.cat(audios, dim=0).float()
        ema_ars = torch.cat(ema_ars, dim=0).float()
        emas = torch.cat(emas, dim=0).float()

        collated_audios.append(audios)
        collated_emas.append(emas)
        collated_ema_ars.append(ema_ars)
    collated_audios = torch.cat(collated_audios, dim=0)
    collated_emas = torch.cat(collated_emas, dim=0)
    collated_ema_ars = torch.cat(collated_ema_ars, dim=0)

    return collated_audios, collated_emas, collated_ema_ars

def newish_car_collate(batch):
    """
    Collates a batch of audio_features and corresponding EMA features in the
    following way:

    For time interval [curr, curr + window_size], we use the audio features
    and pseudolabeled EMA from the same time interval 
    [curr, curr + window_size]. As we are autoregressive w.r.t to the EMA 
    predictions, we use the EMA_ar features from the previous time interval 
    [curr - window_size, curr].

    We then concatenate the audio features, EMA features, EMA_ar features
    independently through all time steps, jumping by window_size.

    Args:
        batch: Batch of tuple - (audio_features, ema_features)

    Returns:
        audio_feats_collated: Collated audio features
        ema_collated: Collated EMA features
        ema_collated_ar: Collated EMA_ar features
    """
    audio_feats, ema = zip(*batch)

    audio_feats = list(audio_feats)
    ema = list(ema)

    for i in range(len(audio_feats)):
        audio_feats[i] = audio_feats[i].detach().cpu()
        ema[i] = ema[i].detach().cpu()
    audio_feats = torch.concatenate(audio_feats, dim=2)
    ema = torch.concatenate(ema, dim=2)


    window_size = 50
    audio_feats_collated = collate(audio_feats, window_size)

    ema_collated = collate(ema, window_size).float()
    ema_collated_ar = ema_collated[:len(ema_collated) - 1]
    first_ema_batch = torch.zeros(1, 12, window_size)

    ema_collated_ar = torch.concatenate([first_ema_batch, ema_collated_ar], dim=0).float()

    return audio_feats_collated, ema_collated, ema_collated_ar

def car_collate(batch):
    """
    Collates a batch of audio_features and corresponding EMA features in the
    following way:

    For time interval [curr, curr + window_size], we use the audio features
    and pseudolabeled EMA from the same time interval 
    [curr, curr + window_size]. As we are autoregressive w.r.t to the EMA 
    predictions, we use the EMA_ar features from the previous time interval 
    [curr - window_size, curr].

    We then concatenate the audio features, EMA features, EMA_ar features
    independently through all time steps, jumping by window_size.

    Args:
        batch: Batch of tuple - (audio_features, ema_features)

    Returns:
        audio_feats_collated: Collated audio features
        ema_collated: Collated EMA features
        ema_collated_ar: Collated EMA_ar features
    """
    audio_feats, ema = zip(*batch)

    audio_feats = list(audio_feats)
    ema = list(ema)

    for i in range(len(audio_feats)):
        audio_feats[i] = audio_feats[i].detach().cpu()
        ema[i] = ema[i].detach().cpu()

    window_size = 50
    step_size = 5
    audio_feats_collated = collate_features(audio_feats, window_size)

    ema_collated = collate_features(ema, window_size).float()
    ema_collated_ar = collate_features(ema, window_size, ar=True).float()
    ema_collated_ar = ema_collated[:len(ema_collated)]
    #first_ema_batch = torch.zeros(1, 12, window_size)

    #ema_collated_ar = torch.concatenate([first_ema_batch, ema_collated_ar], dim=0).float()

    return audio_feats_collated, ema_collated, ema_collated_ar


def collate_features(features: typing.List[torch.FloatTensor], window_size=50, ar=False):
    """
    Performs the autoregressive collating from the car_collate function.
    """
    feats_collated = []
    for feature in features:
        if ar:
            feature = torch.concatenate([torch.zeros(1, 12, 5), feature], dim=2)
        #print(feature.shape)
        collates = [feature[:, :, i:i+window_size] for i in range(0, feature.shape[2] // window_size * window_size, window_size)]
        feature_collated = torch.cat(collates, dim=0)
        feats_collated.append(feature_collated)
    return torch.concatenate(feats_collated, dim=0)

if __name__ == "__main__":
    wav_ema_dataset = WavEMADataset()

    data_loader = torch.utils.data.DataLoader(dataset=wav_ema_dataset, batch_size=1, shuffle=True, collate_fn=car_collate)

    print(next(iter(data_loader)))
