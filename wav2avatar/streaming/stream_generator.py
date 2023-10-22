import asyncio
import sounddevice as sd
import soundfile as sf

class StreamGenerator():

    def __init__(self, wav_path:str = "C:\\Users\\tejas\\Documents\\UCBerkeley\\bci\\wav2avatar\\audios\\mngu0_s1_0001.wav") -> None:
        self.wav_path = wav_path
        self.wav_arr, self.wav_sr = sf.read(self.wav_path)

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

    async def wav_inputs_generator(self, chunksize=1600):
        #sd.play(self.wav_arr, self.wav_sr)
        index = 0
        while index + chunksize <= len(self.wav_arr):
            yield self.wav_arr[index : index + chunksize]
            index += chunksize

if __name__ == "__main__":
    sg = StreamGenerator()
    print(sg.wav_path)