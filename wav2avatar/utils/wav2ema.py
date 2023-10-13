import sounddevice as sd
import soundfile as sf

from tqdm import tqdm
import numpy as np
import librosa
import torchaudio
torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False

import os
import yaml
import torch

from scipy import stats
from queue import Queue

from articulatory.bin.decode import ar_loop
from articulatory.utils import load_model
import s3prl.hub as hub

class MicrophoneStream():
    def __init__(self):
        self.mic_data = np.ndarray([])
        self.q = Queue()
        self.set_batch = lambda indata, frames, time, status: self.q.put(indata.copy().reshape(frames,))
        self.stream = sd.InputStream(samplerate=16000, blocksize=1600, channels=1, callback=self.set_batch)
        self.stream.start()

    def get_next_batch(self, frames=57862):
        print(self.stream.active)
        print(list(self.q.queue))
        return self.q.get()

class EmulatedMicrophoneStream(MicrophoneStream):
    def __init__(self, file):
        super().__init__()
        self.file = file
        self.stream = sf.blocks(file, blocksize=57862)

    def get_next_batch(self):
        return next(self.stream)

class EMAFromMic():
    def __init__(self, model_dir="hprc_no_m1f2_h2emaph_gru_join_nogan_model", gru=True):
        self.gru = gru
        if gru:
            self.input_modality = 'hubert'

            self.hubert_device = 0
            self.model_name = 'hubert_large_ll60k'
            self.hubert_model = getattr(hub, self.model_name)()
            self.hubert_device = 'cuda:%d' % self.hubert_device
            self.hubert_model = self.hubert_model.to(self.hubert_device)

            self.interp_factor = 2
            self.hop_length = 160

            self.inversion_checkpoint_path = f"{model_dir}/best_mel_ckpt.pkl"
            self.inversion_config_path = f"{model_dir}/config.yml"

            with open(self.inversion_config_path) as f:
                self.inversion_config = yaml.load(f, Loader=yaml.Loader)

            if torch.cuda.is_available():
                self.inversion_device = torch.device("cuda:0")
                print("--- using cuda ---")
            else:
                self.inversion_device = torch.device("cpu")

            self.inversion_model = load_model(self.inversion_checkpoint_path, self.inversion_config)
            self.inversion_model.remove_weight_norm()
            self.inversion_model = self.inversion_model.eval().to(self.inversion_device)
        else:
            self.wavlm_device = 0
            self.model_name = 'wavlm_large'
            self.wavlm_model = getattr(hub, self.model_name)()
            self.wavlm_device = 'cuda:%d' % self.wavlm_device
            self.wavlm_model = self.wavlm_model.to(self.wavlm_device)

            self.inversion_checkpoint_path = f"{model_dir}/best_mel_ckpt.pkl"
            self.inversion_config_path = f"{model_dir}/config.yml"

            if torch.cuda.is_available():
                self.inversion_device = torch.device("cuda:0")
                print("--- using cuda ---")
            else:
                self.inversion_device = torch.device("cpu")
            #self.inversion_device = torch.device(f"cuda:0")
            self.inversion_model, self.inversion_config = self.load_model_eval(
                model_dir, device = self.inversion_device, return_config=True)




    def wav2mfcc(wav, sr, num_mfcc=13, n_mels=40, n_fft=320, hop_length=160):
        feat = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        feat = stats.zscore(feat, axis=None)
        return feat

    def wav_to_ema(self, audio):
        if self.gru:
            with torch.no_grad():
                wavs = [torch.from_numpy(audio).float().to(self.hubert_device)]
                #print(wavs)
                states = self.hubert_model(wavs)["hidden_states"]
                feature = states[-1].squeeze(0)
                target_length = len(feature) * self.interp_factor
                feature = torch.nn.functional.interpolate(
                    feature.unsqueeze(0).transpose(1, 2), 
                    size=target_length, 
                    mode='linear', 
                    align_corners=False)
                feature = feature.transpose(1, 2).squeeze(0)  # (seq_len, num_feats)
                feat = feature.to(self.inversion_device)
                if "use_ar" in self.inversion_config["generator_params"] and self.inversion_config["generator_params"]["use_ar"]:
                    pred = ar_loop(self.inversion_model, feat, self.inversion_config, normalize_before=False)
                else:
                    pred = self.inversion_model.inference(feat, normalize_before=False)
                #np.save("test.npy", pred.cpu().numpy())
                return pred.cpu().numpy()

        feat = self.make_wavlm_embeddings(audio)
        with torch.no_grad():
            feat = torch.tensor(feat, dtype=torch.float).to(self.inversion_device)
            if "use_ar" in self.inversion_config["generator_params"] and self.inversion_config["generator_params"]["use_ar"]:
                pred = ar_loop(self.inversion_model, feat, self.inversion_config, normalize_before=False)
            else:
                pred = self.inversion_model.inference(feat, normalize_before=False)

            # (L, H)
            pred = pred[:, 50 : 62] # for ema only
            #np.save(os.path.join(output_dir, get_fid(p)), pred.cpu().numpy())
            return pred.cpu().numpy()
        
    
    def load_config(self, config_dir):
        with open(config_dir) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        return config 

    def get_fid(self, file):
        return os.path.basename(file).split(".")[0]


    def load_model_eval(self, model_dir, device = torch.device("cuda:0"), return_config = False):
        if model_dir[-4:] == ".pkl":
            model_dir = os.path.dirname(model_dir)
        config_file = os.path.join(model_dir, "config.yml")
        ckpt = os.path.join(model_dir, "best_mel_ckpt.pkl")

        config = self.load_config(config_file)
        model = load_model(ckpt, config).to(self.inversion_device)
        print(f"Loaded model parameters from {ckpt}.")
        model.remove_weight_norm()
        model = model.eval()
        if return_config:
            return model, config
        return model

    # Main AAI.
    def hprc_to_ema(self, audio):
        feat = self.make_wavlm_embeddings(audio)
        with torch.no_grad():
            feat = torch.tensor(feat, dtype=torch.float).to(self.inversion_device)
            if "use_ar" in self.inversion_config["generator_params"] and self.inversion_config["generator_params"]["use_ar"]:
                pred = ar_loop(self.inversion_model, feat, self.inversion_config, normalize_before=False)
            else:
                pred = self.inversion_model.inference(feat, normalize_before=False)

            # (L, H)
            pred = pred[:, 50 : 62] # for ema only
            #np.save(os.path.join(output_dir, get_fid(p)), pred.cpu().numpy())
            return pred.cpu().numpy()


    def make_wavlm_embeddings(self, audio):
        #mkdir(output_dir)
        #model_name = 'wavlm_large'
        #model = getattr(hub, model_name)() 
        #device = torch.device(f"cuda:0")
        #model=model.to(device)
        #layer_num = 9

        num_resampled = 0
        num_summed = 0
        wavs = [torch.from_numpy(audio).float().to(self.inversion_device)]
        with torch.no_grad():
            states = self.wavlm_model(wavs)["hidden_states"]
            feature = states[9].squeeze(0)
            #np.save(os.path.join(output_dir, fid), feature.cpu().numpy())
            return feature.cpu().numpy()

if __name__ == "__main__":
    file, sr = sf.read("mngu0_s1_0001.wav")
    print(file.shape)

    mic = MicrophoneStream()
    print("---------starting-------------")
    audio = mic.get_next_batch()
    sf.write("test.wav", audio, 16000)

    model = EMAFromMic("hprc_no_m1f2_h2emaph_gru_joint_nogan_model")
    model.wav_to_ema(audio)