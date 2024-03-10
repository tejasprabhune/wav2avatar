from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
import scipy

import wav2avatar.streaming.hifigan.wav_ema_dataset as wav_ema_dataset
import wav2avatar.streaming.hifigan.collate as collate
from wav2avatar.inversion.linear_inversion import LinearInversion
from wav2avatar.inversion.linear_inversion import EMADataset

from wav2avatar.streaming.emformer.emformer import EMAEmformer

input_dim = 512
num_heads = 16
ffn_dim = 512
num_layers = 15
segment_length = 5
left_context_length = 95
right_context_length = 0

device = 7
emformer = EMAEmformer(
    input_dim=input_dim,
    num_heads=num_heads,
    ffn_dim=ffn_dim,
    num_layers=num_layers,
    segment_length=segment_length,
    left_context_length=left_context_length,
    right_context_length=right_context_length
).to(device)

input_tens = torch.zeros(56, 50, 512).to(device)
lengths = torch.zeros(56,) + 5
lengths = lengths.to(device)

dataset = wav_ema_dataset.WavEMADataset(device=7)

train_amt = int(len(dataset) * 0.9)
test_amt = len(dataset) - train_amt
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_amt, test_amt])

dataloader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=1, 
    shuffle=True)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=1, 
    shuffle=True)

optimizer = torch.optim.Adam(emformer.parameters(), lr=1e-4, betas=[0.5, 0.9], weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.5, milestones=[40000, 80000, 120000, 160000])

def train_emformer_step(batch):
    x = batch[0].to(device).squeeze(0)
    y = batch[1].to(device).squeeze(0)

    x = x.transpose(1, 2)
    lengths = torch.zeros(x.shape[0],) + 5
    lengths = lengths.to(device)
    y_hat = emformer(x, lengths)
    y = y[:, :, :-right_context_length or None]
    loss = F.l1_loss(y_hat, y[:, :, :y_hat.shape[2]])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss

def predict_ema(full_input):
    state = None
    x = None
    lengths = torch.zeros(full_input.shape[0],) + 50
    lengths = lengths.to(device)
    input_size = segment_length + right_context_length
    for i in range(0, full_input.shape[1], segment_length):
        input_ = full_input[:, i:i+input_size, :]
        if input_.shape[1] < input_size:
            input_ = F.pad(input_, (0, 0, 0, input_size - input_.shape[1]))
        x2, lengths, state = emformer.emformer.infer(input_, lengths, state)
        x2 = emformer.output_layer(x2).squeeze(0)
        if x == None:
            x = x2
        else:
            x = torch.cat([x, x2], dim=0)

    x = x.detach().cpu()
    x = EMADataset.butter_bandpass_filter(x, 10, 50)
    return x

def eval_enformer():
    corrs = []
    for i in tqdm(range(10)):
        test_batch = next(iter(test_dataloader))
        test_ema_gt = test_batch[1].squeeze(0).transpose(1, 2).squeeze(0)
        full_input = test_batch[0].squeeze(0).transpose(1, 2)
        full_input = full_input.to(device)

        lengths = torch.zeros(full_input.shape[0],) + 50
        lengths = lengths.to(device)

        x = predict_ema(full_input)
        test_ema_gt = test_ema_gt.detach().cpu()
        test_ema_gt = EMADataset.butter_bandpass_filter(test_ema_gt, 10, 50)

        plot_len = min(x.shape[0], test_ema_gt.shape[0])
        sample_corrs = []
        for i in range(12):
            sample_corrs.append(scipy.stats.pearsonr(x[:plot_len, i], test_ema_gt[:plot_len, i])[0])
        #print(sample_corrs)
        corrs.append(np.mean(sample_corrs))
    return np.mean(corrs)

def save_ckpt(corr):
    torch.save({
        'emformer_state_dict': emformer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, f"ckpts/emf_l{left_context_length}_r{right_context_length}_p{segment_length}_nh{num_heads}__nl{num_layers}_ffd{ffn_dim}_{round(corr, 3)}.pth")

step = 0
eval_step = 10000

data_tqdm = tqdm(dataloader)
loss = torch.tensor([999])

losses = []

best_corr = 0

for batch in data_tqdm:
    data_tqdm.set_description(f"Training step {step}")

    loss = train_emformer_step(batch)
    losses.append(loss.item())

    if step % eval_step == 0:
        corr = eval_enformer()
        print("Correlation:", corr)
        if corr > best_corr:
            best_corr = corr
            save_ckpt(corr)

    data_tqdm.set_postfix(loss=loss.item())
    step += 1