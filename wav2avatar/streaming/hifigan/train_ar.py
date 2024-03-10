import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchaudio

import s3prl
import scipy
import numpy as np
import matplotlib.pyplot as plt

from wav2avatar.streaming.hifigan.wav_ema_dataset import WavEMADataset
from wav2avatar.streaming.hifigan.hifigan_generator import HiFiGANGenerator
import wav2avatar.streaming.hifigan.collate as wav_ema_collate

torch.manual_seed(0)

device = 0
gen = HiFiGANGenerator().to(device)

gen_optimizer = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=[0.5, 0.9], weight_decay=0.0)
gen_scheduler = torch.optim.lr_scheduler.MultiStepLR(gen_optimizer, gamma=0.5, milestones=[40000, 80000, 120000, 160000])

dataset = WavEMADataset()
train_amt = int(len(dataset) * 0.9)
test_amt = len(dataset) - train_amt
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_amt, test_amt])
dataloader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=16, 
    shuffle=True, 
    collate_fn=wav_ema_collate.car_collate)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=100, 
    shuffle=True, 
    collate_fn=wav_ema_collate.car_collate)

feature_model = getattr(s3prl.hub, "wavlm_large")()
feature_model = feature_model.model.feature_extractor.to(device)

window_size = 50

def train_gen_step(batch):
    x = batch[0].to(device)
    y = batch[1].to(device)
    ar = batch[2].to(device)

    y_hat = gen(x, ar)

    ema_loss = F.l1_loss(y_hat, y[:, :, -5:])

    gen_loss = ema_loss

    gen_optimizer.zero_grad()
    gen_loss.backward()
    gen_optimizer.step()
    gen_scheduler.step()

    return gen_loss

def unroll_collated(features):
    return torch.concatenate(list(features), dim=1)

@torch.no_grad()
def eval_gen(batch):
    # Collated features from batch
    clt_audio = batch[0].to(device)
    clt_ema = batch[1].to(device)

    # Add 10 "batches" of blank collated audio as zero padding to start so
    # the generator can autoregressively correctly generate the first 10 frames
    first_audio = torch.zeros((10, 512, 50)).to(device)
    clt_audio = torch.concatenate([first_audio, clt_audio], dim=0)

    # Initialize first EMA CAR to zeros
    first_ema = torch.zeros((1, 12, 50)).to(device)

    # Accumulate all the predicted EMA together and reuse as context
    context_ema = first_ema

    for i in range(clt_audio.shape[0]):
        input_audio = clt_audio[i].unsqueeze(0)
        input_ema = context_ema[:, :, -50:]

        next_gen = gen(input_audio, input_ema)
        context_ema = torch.cat([context_ema, next_gen], dim=2)
    context_ema = context_ema[:, :, 50:]

    # Reformat for plotting purposes
    pred_ema = context_ema.squeeze(0).transpose(1, 0).detach().cpu()
    true_ema = wav_ema_collate.uncollate(clt_ema).squeeze(0).transpose(1, 0).detach().cpu()
    
    return pred_ema, true_ema

def correlations(pred_ema, true_ema):
    corrs = []
    for i in range(12):
        corr = np.corrcoef(pred_ema[i], true_ema[i])[0, 1]
        corrs.append(corr)
    return np.mean(corrs)

@torch.no_grad()
def eval_audio(audio, true_ema):
    ar = torch.zeros((1, 12, window_size)).to(device)

    audio_feats = feature_model(audio.to(device))
    print(audio_feats.shape)

    collated_audio = wav_ema_collate.collate_features(audio_feats.unsqueeze(0))
    print("collated_audio:", collated_audio.shape)

    pred = []
    for audio_feat in collated_audio:
        #print(audio_feat.shape, ar.shape)
        pred.append(gen(audio_feat.unsqueeze(0), ar)[:, :, :window_size])
        ar = pred[-1][:, :, :window_size]
    full_pred = torch.concatenate(pred, dim=2).squeeze(0)
    #print(full_pred.shape)
    full_pred = full_pred.transpose(1, 0).cpu().numpy()

    print(full_pred.shape, true_ema.shape)
    return full_pred

def save_ckpt(corr, train_split=90):
    torch.save({
        'gen_state_dict': gen.state_dict(),
        'gen_optimizer_state_dict': gen_optimizer.state_dict(),
        'gen_scheduler_state_dict': gen_scheduler.state_dict(),
    }, f"ckpts/hfgen_l1_noup_5res_{train_split}_{round(corr, 2)}.pth")

step = 0
disc_start = 0
gen_start = 1

eval_step = 200

# (1) (2) Get batch
data_tqdm = tqdm(dataloader)
gen_loss = torch.tensor([999])

test_batch = next(iter(test_dataloader))

best_corr = 0

for batch in data_tqdm:
    data_tqdm.set_description(f"Training step {step}")

    if step >= gen_start:
        gen_loss = train_gen_step(batch)

    if step % eval_step == 0:
        pred_ema, true_ema = eval_gen(test_batch)
        corr = correlations(pred_ema, true_ema)
        print(f"Correlations: {round(corr, 2)}")
        feat_num = 0
        plt.plot(true_ema[:300, feat_num], label="target")
        plt.plot(pred_ema[:300, feat_num], label="pred")
        plt.legend()
        plt.show()

        if corr > best_corr:
            save_ckpt(corr)
            best_corr = corr

    data_tqdm.set_postfix(gen_loss=gen_loss.item())
    step += 1

pred_ema, true_ema = eval_gen(test_batch)
corr = correlations(pred_ema, true_ema)
print(f"Correlations: {round(corr, 2)}")

if corr > best_corr:
    save_ckpt(corr)
    best_corr = corr