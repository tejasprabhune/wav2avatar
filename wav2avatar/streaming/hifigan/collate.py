import torch
import torch.utils.data

from wav2avatar.streaming.hifigan.wav_ema_dataset import WavEMADataset

import typing

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
    audio_feats_collated = collate_features(audio_feats, window_size)

    ema_collated = collate_features(ema, window_size).float()
    ema_collated_ar = ema_collated[:len(ema_collated) - 1]
    first_ema_batch = torch.zeros(1, 12, window_size)

    ema_collated_ar = torch.concatenate([first_ema_batch, ema_collated_ar], dim=0).float()

    return audio_feats_collated, ema_collated, ema_collated_ar

def collate_features(features: typing.List[torch.FloatTensor], window_size=50):
    """
    Performs the autoregressive collating from the car_collate function.
    """
    feats_collated = []
    for feature in features:
        collates = [feature[:, :, i:i+window_size] for i in range(0, feature.shape[2] // window_size * window_size, window_size)]
        feature_collated = torch.cat(collates, dim=0)
        feats_collated.append(feature_collated)
    return torch.concatenate(feats_collated, dim=0)

if __name__ == "__main__":
    wav_ema_dataset = WavEMADataset()

    data_loader = torch.utils.data.DataLoader(dataset=wav_ema_dataset, batch_size=1, shuffle=True, collate_fn=car_collate)

    print(next(iter(data_loader)))