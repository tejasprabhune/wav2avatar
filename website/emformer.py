import scipy
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.models import Emformer
import numpy as np
import webrtcvad

class EMAEmformer(torch.nn.Module):

    def __init__(
        self,
        input_dim=512,
        num_heads=8,
        ffn_dim=256,
        num_layers=15,
        segment_length=5,
        left_context_length=45,
        right_context_length=0
    ):
        super().__init__()

        self.emformer = Emformer(
            input_dim=input_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            segment_length=segment_length,
            left_context_length=left_context_length,
            right_context_length=right_context_length
        )

        self.output_layer = torch.nn.Linear(512, 12)

        self.vad = webrtcvad.Vad(1)
    
    def forward(self, x, lengths=None):
        x, lengths = self.emformer(x, lengths)
        x = self.output_layer(x)
        x = x.transpose(2, 1)
        return x
    
    def infer(self, x, lengths, state):
        x, lengths, state = self.emformer.infer(x, lengths, state)
        x = self.output_layer(x)
        x = x.transpose(2, 1)
        return x, state

    def predict_ema(self, full_input, state):
        full_input = full_input.transpose(1, 2)

        x = None
        lengths = torch.zeros(full_input.shape[0],) + 50
        lengths = lengths.to(0)
        for i in range(0, full_input.shape[1], 5):
            input_ = full_input[:, i:i+5, :]
            if input_.shape[1] < 5:
                input_ = F.pad(input_, (0, 0, 0, 5 - input_.shape[1]))
            x2, lengths, state = self.emformer.infer(input_, lengths, state)
            x2 = self.output_layer(x2).squeeze(0)
            if x == None:
                x = x2
            else:
                x = torch.cat([x, x2], dim=0)

        x = x.detach().cpu()
        x = torch.tensor(EMAEmformer.butter_bandpass_filter(x, 10, 50).copy())
        return x, state

    def predict_ema(self, full_input, state=None):
        x = None
        segment_length = 5
        device = 0
        lengths = torch.zeros(full_input.shape[0],) + 50
        lengths = lengths.to(device)
        input_size = segment_length #+ right_context_length
        for i in range(0, full_input.shape[1], segment_length):
            input_ = full_input[:, i:i+input_size, :]
            if input_.shape[1] < input_size:
                input_ = F.pad(input_, (0, 0, 0, input_size - input_.shape[1]))
            x2, lengths, state = self.emformer.infer(input_, lengths, state)
            x2 = self.output_layer(x2).squeeze(0)
            if x == None:
                x = x2
            else:
                x = torch.cat([x, x2], dim=0)

        x = x.detach().cpu()
        x = torch.tensor(EMAEmformer.butter_bandpass_filter(x, 10, 50).copy())
        return x, state

    def butter_bandpass(cut, fs, order=5):
        if isinstance(cut, list) and len(cut) == 2:
            return scipy.signal.butter(order, cut, fs=fs, btype="bandpass")
        else:
            return scipy.signal.butter(order, cut, fs=fs, btype="low")

    def butter_bandpass_filter(data, cut, fs, order=5):
        b, a = EMAEmformer.butter_bandpass(cut, fs, order=order)
        y = scipy.signal.filtfilt(b, a, data, padlen=len(data) - 1, axis=0)
        return y

    def float_to_pcm16(self, audio):
        ints = (audio * 32768).astype(np.int16)
        little_endian = ints.astype("<u2")
        buf = little_endian.tobytes()
        return buf

    def is_speech(self, audio, sr):
        speech_segments = []
        for i in range(0, len(audio), sr // 100):
            if self.vad.is_speech(
                self.float_to_pcm16(audio[i : i + sr // 100]), sr
            ):
                speech_segments.append(audio[i : i + sr // 100])
        print(audio.shape, len(speech_segments))
        return len(speech_segments) >= 4