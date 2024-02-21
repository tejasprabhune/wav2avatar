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
from wav2avatar.streaming.hifigan.hifigan_discriminators import HiFiGANMultiScaleMultiPeriodDiscriminator
from wav2avatar.streaming.hifigan.hifigan_generator import HiFiGANGenerator
import wav2avatar.streaming.hifigan.collate as wav_ema_collate

from wav2avatar.streaming.hifigan.losses import GeneratorAdversarialLoss, DiscriminatorAdversarialLoss, FeatureMatchLoss

torch.manual_seed(0)

device = 0
disc = HiFiGANMultiScaleMultiPeriodDiscriminator().to(device)
gen = HiFiGANGenerator(
    in_channels=512, 
    out_channels=12, 
    ar_input=600, 
    use_tanh=False,
    use_mlp_ar=False,
    resblock_kernel_sizes=(3, 7, 11, 15, 19),
    resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5), (1, 3, 5), (1, 3, 5)],
                       ).to(device)

disc_optimizer = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=[0.5, 0.9], weight_decay=0.0)
disc_scheduler = torch.optim.lr_scheduler.MultiStepLR(disc_optimizer, gamma=0.5, milestones=[40000, 80000, 120000, 160000])

gen_optimizer = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=[0.5, 0.9], weight_decay=0.0)
gen_scheduler = torch.optim.lr_scheduler.MultiStepLR(gen_optimizer, gamma=0.5, milestones=[40000, 80000, 120000, 160000])

disc_adv_loss = DiscriminatorAdversarialLoss()
gen_adv_loss = GeneratorAdversarialLoss()
feat_match_loss = FeatureMatchLoss()

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

def train_disc_step(batch):
    x = batch[0].to(device)
    y = batch[1].to(device)
    ar = batch[2].to(device)
    #print(f"Loaded batch with shapes x: {x.shape}, y: {y.shape}, ar: {ar.shape}")

    # (3) Get predictions
    with torch.no_grad():
        y_hat = gen(x, ar)
    p = disc(y)
    p_hat = disc(y_hat)

    # (4) Calculate loss
    real_loss, fake_loss = disc_adv_loss(p_hat, p)
    disc_loss = real_loss + fake_loss
    #print(f"Discriminator loss: {disc_loss}")

    # (5) Backprop
    disc_optimizer.zero_grad()
    disc_loss.backward()
    disc_optimizer.step()
    disc_scheduler.step()

    return disc_loss

def train_gen_step(batch):
    x = batch[0].to(device)
    y = batch[1].to(device)
    ar = batch[2].to(device)

    y_hat = gen(x, ar)
    p_hat = disc(y_hat)
    p = disc(y)

    adv_loss = gen_adv_loss(p_hat)

    feat_loss = feat_match_loss(p_hat, p)

    ema_loss = F.l1_loss(y_hat[:, :, :window_size], y)

    gen_loss = adv_loss + 2 * feat_loss + 45 * ema_loss

    gen_optimizer.zero_grad()
    gen_loss.backward()
    gen_optimizer.step()
    gen_scheduler.step()

    return gen_loss

def unroll_collated(features):
    return torch.concatenate(list(features), dim=1)

@torch.no_grad()
def eval_gen(batch):
    x = batch[0].to(device)
    y = batch[1].to(device)
    y_unrolled = unroll_collated(y)

    ar = torch.zeros((1, 12, window_size)).to(device)
    pred = []
    for audio_feat in x:
        #print(audio_feat.shape, ar.shape)
        pred.append(gen(audio_feat.unsqueeze(0), ar)[:, :, :window_size])
        ar = pred[-1][:, :, :window_size]
    full_pred = torch.concatenate(pred, dim=2).squeeze(0)
    #print(full_pred.shape)
    full_pred = full_pred.transpose(1, 0).cpu().numpy()
    y_unrolled = y_unrolled.transpose(1, 0).cpu().numpy()
    #print(full_pred.shape, y_unrolled.shape)
    #print("First feature corr:", scipy.stats.pearsonr(full_pred[:, 0], y_unrolled[:, 0]))
    corrs = []
    for i in range(12):
        corrs.append(scipy.stats.pearsonr(full_pred[:, i], y_unrolled[:, i])[0])
    corr_mean = np.mean(corrs)
    print("Correlations:", corr_mean)
    return full_pred, y_unrolled, corr_mean

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
        'disc_state_dict': disc.state_dict(),
        'gen_optimizer_state_dict': gen_optimizer.state_dict(),
        'disc_optimizer_state_dict': disc_optimizer.state_dict(),
        'gen_scheduler_state_dict': gen_scheduler.state_dict(),
        'disc_scheduler_state_dict': disc_scheduler.state_dict()
    }, f"ckpts/hfcar_l1_noup_5res_arconv3_{train_split}_{round(corr, 2)}.pth")

step = 0
disc_start = 0
gen_start = 1

eval_step = 200

# (1) (2) Get batch
data_tqdm = tqdm(dataloader)
gen_loss = torch.tensor([999])
disc_loss = torch.tensor([999])

test_batch = next(iter(test_dataloader))

best_corr = 0

for batch in data_tqdm:
    data_tqdm.set_description(f"Training step {step}")

    if step >= gen_start:
        gen_loss = train_gen_step(batch)

    if step >= disc_start and step % 5 == 0:
        disc_loss = train_disc_step(batch)
    
    if step % eval_step == 0:
        pred_ema, true_ema, corr = eval_gen(test_batch)
        feat_num = 0
        plt.plot(true_ema[:300, feat_num], label="target")
        plt.plot(pred_ema[:300, feat_num], label="pred")
        plt.legend()
        plt.show()

        if corr > best_corr:
            save_ckpt(corr)
            best_corr = corr

    data_tqdm.set_postfix(gen_loss=gen_loss.item(), disc_loss=disc_loss.item())
    step += 1

pred_ema, true_ema, corr = eval_gen(test_batch)
feat_num = 0

if corr > best_corr:
    save_ckpt(corr)
    best_corr = corr