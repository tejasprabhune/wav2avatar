from tqdm import tqdm

import torch
import torch.nn.functional as F

import wav2avatar.streaming.hifigan.wav_ema_dataset
import wav2avatar.streaming.hifigan.hifigan_generator
import wav2avatar.streaming.hifigan.hifigan_discriminators

class Trainer():
    def __init__(
            self,
            steps,
            epochs,
            data_loader,
            sampler,
            model,
            criterion,
            optimizer,
            scheduler,
            config,
            device
            ):
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.sampler = sampler
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

    def run(self):

        self.tqdm = tqdm(initial=0, total=self.steps, desc="[train]")

        while True:
            self._train_epoch()

            if self.finish_train:
                break
    
        self.tqdm.close()
    
    def _train_epoch(self):
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            self._train_step(batch)

            if self.finish_train:
                break
        
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch