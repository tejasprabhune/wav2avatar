import os
import yaml
import torch
import librosa
import argparse
import torchaudio
torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False

import numpy as np
import s3prl.hub as hub

from scipy import stats
from articulatory.bin.decode import ar_loop
from articulatory.utils import load_model

class Wav2EMA():
    """
    Sets up speech to EMA and speech to MRI inversion models (WIP):
    
    1. BiGRU implementation from Wu et al. 2023
    (https://ieeexplore.ieee.org/document/10096796)

    2. Transformer EMA inversion implementation by Bohan Yu

    3. Transformer MRI inversion implementation by Prabhune et al. (Bohan Yu)
    (https://speech-avatar.github.io/multimodal-mri-avatar/)
    """
    def __init__(self, model_dir="hprc_no_m1f2_h2emaph_gru_join_nogan_model", gru=True, mri=False):
        self.mri = mri
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
        with torch.no_grad():
            wavs = [torch.from_numpy(audio).float().to(self.hubert_device)]
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

    def hprc_to_ema(self, audio):
        """wav2ema for Transformer inversion."""
        feat = self.make_wavlm_embeddings(audio)
        with torch.no_grad():
            feat = torch.tensor(feat, dtype=torch.float).to(self.inversion_device)
            if "use_ar" in self.inversion_config["generator_params"] and self.inversion_config["generator_params"]["use_ar"]:
                pred = ar_loop(self.inversion_model, feat, self.inversion_config, normalize_before=False)
            else:
                pred = self.inversion_model.inference(feat, normalize_before=False)

            # (L, H)
            if self.mri:
                pred = pred[:, -230:]
            else:
                pred = pred[:, 50 : 62] # for ema only
            #np.save(os.path.join(output_dir, get_fid(p)), pred.cpu().numpy())
            return pred.cpu().numpy()

    def make_wavlm_embeddings(self, audio):
        num_resampled = 0
        num_summed = 0
        wavs = [torch.from_numpy(audio).float().to(self.inversion_device)]
        with torch.no_grad():
            states = self.wavlm_model(wavs)["hidden_states"]


            # (L, H)
            feature = states[9].squeeze(0)
            #np.save(os.path.join(output_dir, fid), feature.cpu().numpy())
            if self.mri:
                feature_interpolated = torch.nn.functional.interpolate(feature.cpu().unsqueeze(0).transpose(1, 2),
                                                        int(feature.shape[0] / 50 * (20000 / 240)),
                                                        mode='linear',
                                                        align_corners=False )
                feature_interpolated = feature_interpolated.squeeze(0).transpose(0, 1).numpy()
                return feature_interpolated
            return feature.cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invert audio to EMA.")

    parser.add_argument('--model_dir', dest="model_dir", action="store",
                        help="Directory of inversion model", 
                        default="""C:/Users/tejas/Documents/UCBerkeley/bci/SpectrogramSynthesis/hprc_no_m1f2_wlm2tvph_norm_transformer_conv_joint_nogan_v5/""")
    parser.add_argument('--wav_file', dest="wav_file", action="store",
                        help="Path of input wav file to invert")
    parser.add_argument('--save_dir', dest="save_dir", action="store",
                        help="Directory of where to store inverted EMA npy")


    args = parser.parse_args()

    args.model_dir = (
    "C:/Users/tejas/Documents/UCBerkeley/bci/Spectrogram"
    + "Synthesis/mri_timit_230_wlm2f0_mri_230_pretrain_mri_ema/")

    audio, sr = librosa.load(f"{args.wav_file}", sr=16000)
    wav_name = os.path.basename(args.wav_file)
    print(f"Loaded {wav_name} at sample rate {sr} and shape {audio.shape}.\n")

    model = Wav2EMA(model_dir=args.model_dir, gru=False, mri=True)
    print("Loaded inversion model.\n")

    ema = model.hprc_to_ema(audio)
    print("Converted audio to EMA.\n")
    
    save_file = f"{args.save_dir}/{wav_name[:-4]}.npy"
    np.save(save_file, ema)
    print(f"Saved EMA at {save_file}.")