import torch
import torch.utils.data

from wav2avatar.streaming.hifigan.wav_ema_dataset import WavEMADataset

import typing

def car_collate(batch):
    audio_feats, ema = zip(*batch)

    audio_feats = list(audio_feats)
    ema = list(ema)

    window_size = 50
    audio_feats_collated = collate_features(audio_feats, window_size)
    ema_collated = collate_features(ema, window_size)
    ema_collated = ema_collated[:len(ema_collated) - 2]
    first_ema_batch = torch.zeros(1, 12, window_size)

    return audio_feats, ema

def collate_features(features: typing.List[torch.FloatTensor], window_size=50):
    feats_colllated = []
    for feature in features:
        collates = [feature[:, :, i:i+window_size] for i in range(0, feature.shape[2] // window_size * window_size, window_size)]
        feature_collated = torch.cat(collates, dim=0)
        feats_colllated.append(feature_collated)
    return torch.concatenate(feats_colllated, dim=0)



if __name__ == "__main__":

    wav_ema_dataset = WavEMADataset()

    data_loader = torch.utils.data.DataLoader(dataset=wav_ema_dataset, batch_size=1, shuffle=True, collate_fn=car_collate)

    print(next(iter(data_loader)))