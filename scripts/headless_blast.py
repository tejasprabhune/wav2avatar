from wav2avatar.utils import MayaUtils
import argparse
import pathlib
import numpy as np

parser = argparse.ArgumentParser(description="Headless Maya Avatar Playblasts")
parser.add_argument("--maya_file", type=str, help="Path to Maya file")
parser.add_argument("--ema", type=str, help="Path to EMA data")
parser.add_argument("--wav", type=str, help="Path to audio file")
parser.add_argument("--outdir", type=str, help="Output directory")

args = parser.parse_args()

outdir = pathlib.Path(args.outdir).resolve()
maya_file = pathlib.Path(args.maya_file).resolve()
wav_file = pathlib.Path(args.wav).resolve()
ema_file = pathlib.Path(args.ema).resolve()

ema = np.load(ema_file)
num_frames = ema.shape[0]

MayaUtils.open_standalone()
import maya.cmds as cmds
MayaUtils.open_file(maya_file)
MayaUtils.set_side_cam()
wav_node = MayaUtils.import_wav(wav_file)
MayaUtils.animate_web_avatar(ema_file)
MayaUtils.playblast_avatar(start=0, end=num_frames, width=1920, height=1080, filename=outdir / wav_file.stem, sound_node=wav_node)