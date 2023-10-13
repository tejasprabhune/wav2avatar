"""
TODO:
1. read wav
2. batch generator
3. send_batch
"""

from typing import List

import os
import sys
import time
import pickle
import socket
import asyncio
import webrtcvad
import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt

from collections import deque
from multiprocessing import shared_memory

from wav2avatar.utils.nema_data import NEMAData
from wav2avatar.utils.wav2ema import EMAFromMic



class Wav2Maya:
    def __init__(
        self,
        wav_path: str = "audios/mngu0_s1_0001.wav",
        port: int = 5005,
        parts: List[str] = ["tt", "tb", "td", "li", "ll", "ul"],
    ) -> None:
        self.wav_path = wav_path
        self.parts = parts

        self.log("reading wav file")
        self.wav_arr, self.wav_sr = sf.read(self.wav_path)
        self.vad = webrtcvad.Vad(1)

        self.log("allocating shared memory")
        self.shm = shared_memory.SharedMemory(
            name="audio_stream", 
            create=True, 
            size=5000000
        )

        self.log("importing model")
        #model_dir = (
        #    "C:/Users/tejas/Documents/UCBerkeley/bci/Spectrogram"
        #    + "Synthesis/hprc_no_m1f2_h2emaph_gru_joint_nogan_model/"
        #)
        #self.ema_handler = EMAFromMic(model_dir=model_dir)

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

        self.current_frame = 0

        #zeros_arr = np.zeros((1600,))
        self.ema_handler.wav_to_ema(self.wav_arr[:1600])

        # Timed test lists:
        self.model_times = []
        self.send_times = []

    async def wav_inputs_generator(self, chunksize=1600):
        sd.play(self.wav_arr, self.wav_sr)
        index = 0
        while index + chunksize <= len(self.wav_arr):
            yield self.wav_arr[index : index + chunksize]
            index += chunksize

    async def inputstream_generator(self, channels=1):
        q_in = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def callback(indata, frame_count, time_info, status):
            loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

        stream = sd.InputStream(samplerate=16000, callback=callback, channels=channels, blocksize=1600)
        print("\n\n ----------listening--------- \n\n")
        with stream:
            while True:
                indata, status = await q_in.get()
                yield indata[:, 0]

    def stream_to_memory(self):
        self.log("sharing to memory")
        send_start = time.time()
        print(len(self.cumulative_ema))
        ser_message = pickle.dumps(self.cumulative_ema, protocol=4)
        self.shm.buf[:len(ser_message)] = ser_message
        print(len(ser_message))
        send_end = time.time()

        self.send_times.append(send_end - send_start)

    def batch_to_ema(self, batch):
        """Returns maya_data (dict) of 10 frame data for each part"""
        self.log("converting batch to ema")

        self.cumulative_audio = self.update_cumulative(
            self.cumulative_audio, batch
        )
        
        output_frames = None

        if self.is_speech(batch, self.wav_sr):

            model_start = time.time()
            last_second_ema = self.ema_handler.wav_to_ema(
                self.cumulative_audio[-16000:]
            )
            last_second_nema = NEMAData(ema_data=last_second_ema, is_file=False)

            last_second_nema.offset_parts(["li"])

            for part in self.parts:
                last_second_nema.maya_data[part] = last_second_nema.maya_data[part][-10:]

                self.last_ema_frame[part][0] = last_second_nema.maya_data[part][-1:][0]
            
            output_frames = last_second_nema.maya_data

            model_end = time.time()
            
        else:
            model_start = time.time()

            frames_data = {}
            for part in self.parts:
                frames_data[part] = []
                for _ in range(10):
                    frames_data[part].append(self.last_ema_frame[part][0])
            output_frames = frames_data
            model_end = time.time()
        self.model_times.append(model_end - model_start)
        return output_frames
    
    async def input_to_ema(self):
        self.log("listening")
        async for batch in self.inputstream_generator():
        #async for batch in self.wav_inputs_generator():
            print(f"batch: {self.current_frame}")
            message = [self.current_frame, self.batch_to_ema(batch)]
            #self.send_message([self.current_frame, message])
            self.cumulative_ema.append(message)
            self.stream_to_memory()
            self.current_frame += 1
        
    async def run(self):
        try:
            await asyncio.wait_for(self.input_to_ema(), timeout=15)
        except asyncio.TimeoutError:
            pass

    def interpolate_batch(prev_batch, curr_batch):
        cubic_bez = lambda arr, t: ((1 - t) ** 3) * arr[0] + 3 * ((1 - t) ** 2) * t * arr[1] + 3 * (1 - t) * (t ** 2) * arr[2] + (t ** 3) * arr[3]

        interp_batch = []
        interp_batch.append(prev_batch[-1])
        interp_batch.extend(curr_batch[:3])

        ys = [cubic_bez(interp_batch, t / 3) for t in range(0, 4)]

        interpolated_next_batch = []
        interpolated_next_batch.extend(ys[1:]) # ignore last frame from prev batch
        interpolated_next_batch.extend(curr_batch[3:])

        return interpolated_next_batch

    def update_cumulative(self, cumulative, update):
        if not len(cumulative):
            cumulative = update
        else:
            cumulative = np.append(cumulative, update, axis=0)
        return cumulative

    def float_to_pcm16(self, audio):
        ints = (audio * 32768).astype(np.int16)
        little_endian = ints.astype('<u2')
        buf = little_endian.tobytes()
        return buf

    def is_speech(self, audio, sr):
        speech_segments = []
        for i in range(0, len(audio), sr // 100):
            if self.vad.is_speech(
                self.float_to_pcm16(audio[i : i + sr // 100]), sr
            ):
                speech_segments.append(audio[i : i + sr // 100])
        return len(speech_segments) >= 9

    def log(self, message):
        print(f"\n--- {message} ---")

    def save_model_times_fig(self):
        plt.figure(1, figsize=(8, 6))
        plt.scatter([x for x in range(0, len(self.model_times))], 
                    self.model_times, label="model")
        plt.scatter([x for x in range(0, len(self.send_times))], 
                    self.send_times, label="send to maya")
        plt.legend()
        plt.yticks(np.arange(0, 0.3, 0.05))
        plt.xlabel("batch")
        plt.ylabel("time (sec)")
        plt.title("Overall model/silence processing time per 10 frames")
        plt.savefig('times/stream_mngu0_model.png')


if __name__ == "__main__":
    sender = Wav2Maya(wav_path="audios/mngu0_s1_0008.wav",port=5010)

    try:
        asyncio.run(sender.run())
    except KeyboardInterrupt:
        sys.exit(0)
    
    sender.cumulative_ema = []
    for i in range(2000):
        sender.cumulative_ema.append([i, sender.last_ema_frame])
    sender.stream_to_memory()
    time.sleep(3)

    with open('times/stream_mngu0_model.txt', 'w') as f:
        f.write(str(sender.model_times))
    with open('times/stream_mngu0_send.txt', 'w') as f:
        f.write(str(sender.send_times))
    sf.write("audios/audio_stream.wav", sender.cumulative_audio, sender.wav_sr)
    #sender.save_model_times_fig()