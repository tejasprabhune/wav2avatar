import PySide2.QtGui
import maya.cmds as cmds
from maya import OpenMayaUI as omui
import maya.api.OpenMaya as om
from maya.api.OpenMayaAnim import MAnimControl as mac
from shiboken2 import wrapInstance
from PySide2 import QtUiTools, QtCore, QtGui, QtWidgets
from functools import partial
import sys
import asyncio

from collections import deque
import numpy as np
import soundfile as sf
import webrtcvad
import torch

from wav2avatar.streaming.stream_generator import StreamGenerator
from wav2avatar.utils.utils import Utils
from wav2avatar.utils.nema_data import NEMAData
from wav2avatar.utils.wav2ema import EMAFromMic

class AvatarUI(QtWidgets.QWidget):

    window = None

    def __init__(self, 
                 wav_path="C:\\Users\\tejas\\Documents\\UCBerkeley\\bci\\wav2avatar\\audios\\mngu0_s1_0001.wav", 
                 parts=["tt", "tb", "td", "li"], 
                 parent = None):

        super().__init__(parent=parent)

        self.stream_generator = StreamGenerator()
        self.wav_gen = self.stream_generator.wav_inputs_generator
        self.input_gen = self.stream_generator.inputstream_generator

        self.wav_path = wav_path
        print(wav_path)
        self.parts = parts

        Utils.log("reading wav file")
        self.wav_arr, self.wav_sr = sf.read(self.wav_path)
        self.vad = webrtcvad.Vad(1)

        # Import Bohan's transformer inversion model
        model_dir = (
            "C:/Users/tejas/Documents/UCBerkeley/bci/Spectrogram"
            + "Synthesis/hprc_no_m1f2_wlm2tvph_norm_transformer_conv_joint_nogan_v5/")
        self.ema_handler = EMAFromMic(model_dir=model_dir, gru=False)

        self.cumulative_audio = []
        self.last_ema_frame = {
            "tt": [[0, 0, -4.196, 8.963]],
            "tb": [[0, 0, -3.058, -0.644]],
            "td": [[0, 0, -5.57, -10.255]],
            "li": [[0, 0, 0.151, 19.762]],
            "ll": [[0, 0, -2.147, 13.173]],
            "ul": [[0, 0, 0, 13.554]]
        }
        #self.cumulative_ema = []
        self.cumulative_ema = deque(maxlen=20)
        self.cumulative_npy = np.array([]).reshape(0, 12)

        self.current_frame = 0

        #zeros_arr = np.zeros((1600,))
        self.ema_handler.wav_to_ema(self.wav_arr[:1600])
        self.first_npy_frame = self.ema_handler.wav_to_ema(self.wav_arr[:1600])[:, 0:12]

        # Qt loads UI from path and sets Maya as path (based on my understanding)
        self.setWindowFlags(QtCore.Qt.Window)
        self.widget_path = ('C:\\Users\\tejas\\Documents\\UCBerkeley\\' +
                            'bci\\wav2avatar\\wav2avatar\\plugin_qt\\' +
                            'plugin_ui\\')
        self.widget = QtUiTools.QUiLoader().load(self.widget_path + 'wav2avatar_ui.ui')
        self.widget.setParent(self)
        
        self.resize(500, 300)

        # Sets up the two buttons and their 'click' functionality
        self.btn_close = self.widget.findChild(QtWidgets.QPushButton, 
                                               'btn_close')
        self.btn_close.clicked.connect(self.close_window)

        self.btn_invert = self.widget.findChild(QtWidgets.QPushButton,
                                               'btn_invert')
        self.btn_invert.clicked.connect(self.click_invert)
    
    def resizeEvent(self, event) -> None:
        self.widget.resize(self.width(), self.height())
    
    def close_window(self):
        self.ema_handler.clear_cache()
        del self.ema_handler
        torch.cuda.empty_cache()
        self.destroy()

    def click_invert(self):
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            sys.exit(0)
        
    def batch_to_ema(self, batch):
        Utils.log("converting batch to ema")

        # Adds new batch to cumulative audio
        self.cumulative_audio = Utils.update_cumulative(
            self.cumulative_audio, batch
        )

        output_frames = None

        if Utils.is_speech(self.vad, batch, self.wav_sr):
            last_second_ema = self.ema_handler.wav_to_ema(
                self.cumulative_audio[-32000:]
            )

            second_last_batch = last_second_ema[-20:-10, 0:12]

            if len(second_last_batch) < 1:
                return 0
            
            Utils.parts_interpolate_batch(self.cumulative_npy, second_last_batch)

            self.cumulative_npy = np.vstack(
                [self.cumulative_npy, second_last_batch]
            )
            last_second_nema = NEMAData(
                ema_data=last_second_ema, is_file=False
            )

            last_second_nema.offset_parts(["li", "ll", "ul"])

            for part in self.parts:
                last_second_nema.maya_data[part] = last_second_nema.maya_data[
                    part
                ][-10:]
                self.last_ema_frame[part][0] = last_second_nema.maya_data[
                    part
                ][-1:][0]

            output_frames = last_second_nema.maya_data
        else:
            if self.cumulative_npy.shape == (0, 12):
                self.cumulative_npy = self.first_npy_frame
            else:
                self.cumulative_npy = np.vstack(
                    [self.cumulative_npy, self.cumulative_npy[-10:]]
                )

            frames_data = {}
            for part in self.parts:
                frames_data[part] = []
                for _ in range(10):
                    frames_data[part].append(self.last_ema_frame[part][0])
            output_frames = frames_data
        print(self.cumulative_npy.shape)
        self.prev_batch = batch
        return output_frames
    
    async def input_to_ema(self):
        async for batch in self.wav_gen():

            message = [self.current_frame, self.batch_to_ema(batch)]
            if type(message[1]) == dict:
                self.animate_mouth(message[1], 0)

            self.current_frame += 1
    
    async def run(self):
        try:
            await asyncio.wait_for(self.input_to_ema(), timeout=15)
        except asyncio.TimeoutError:
            pass

    axes = ["X", "Y", "Z"]
    def key_translate(self, axis:int, mesh:str, key:int, value:float):
        cmds.setKeyframe(
            mesh,
            time=key,
            attribute=f"translate{self.axes[axis]}",
            value=value,
        )

    def clear_keys(self, mesh):
        cmds.cutKey(mesh, time=(0, 1000), attribute="translateX")
        cmds.cutKey(mesh, time=(0, 1000), attribute="translateY")
        cmds.cutKey(mesh, time=(0, 1000), attribute="translateZ")

    def get_value(self, key, axis):
        if type(key[axis + 1]) in [int, float]:
            value = key[axis + 1]
        elif type(key[axis + 1]) == np.float32:
            value = key[axis + 1].item()
        return value

    def animate_mouth(self, maya_data, last_frame):
        for part in self.parts:
            mesh = f"{part}Handle"
            self.clear_keys(mesh)
        maya_range = len(maya_data["li"])
        for i in range(maya_range):
            for part in self.parts:
                key = maya_data[part][i]
                x_value = self.get_value(key, 2)
                y_value = self.get_value(key, 1)
                mesh = f"{part}Handle"

                self.key_translate(2, mesh, i + last_frame, x_value)
                self.key_translate(1, mesh, i + last_frame, y_value)
        mac.setMinMaxTime(om.MTime(0), om.MTime(last_frame + maya_range))
        mac.setAnimationStartEndTime(om.MTime(0), om.MTime(last_frame + maya_range))
        mac.setCurrentTime(om.MTime(last_frame))
        mac.setPlaybackMode(0)
        mac.playForward()

def openWindow():
    if QtWidgets.QApplication.instance():
        for win in (QtWidgets.QApplication.allWindows()):
            if 'wav2avatar_window' in win.objectName():
                win.destroy()
    
    mayaMainWindowPtr = omui.MQtUtil.mainWindow()
    mayaMainWindow = wrapInstance(int(mayaMainWindowPtr), QtWidgets.QWidget)
    AvatarUI.window = AvatarUI(parent=mayaMainWindow)
    AvatarUI.window.setObjectName('wav2avatar_window')
    AvatarUI.window.setWindowTitle('wav2avatar')
    AvatarUI.window.show()

openWindow()