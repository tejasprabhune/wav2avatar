import numpy as np

from wav2avatar.utils.utils import Utils
from wav2avatar.utils.nema_data import NEMAData

class Wav2Batches():

    def batch_to_ema(batch, 
                     cumulative_audio, 
                     vad, 
                     wav_sr, 
                     ema_handler,
                     cumulative_npy, 
                     parts,
                     last_ema_frame,
                     first_npy_frame):
        Utils.log("converting batch to ema")

        # Adds new batch to cumulative audio
        cumulative_audio = Utils.update_cumulative(
            cumulative_audio, batch
        )

        output_frames = None

        if Utils.is_speech(vad, batch, wav_sr):
            last_second_ema = ema_handler.wav_to_ema(
                cumulative_audio[-32000:]
            )

            second_last_batch = last_second_ema[-20:-10, 0:12]

            if len(second_last_batch) < 1:
                return 0
            
            Utils.parts_interpolate_batch(cumulative_npy, second_last_batch)

            cumulative_npy = np.vstack(
                [cumulative_npy, second_last_batch]
            )
            last_second_nema = NEMAData(
                ema_data=last_second_ema, is_file=False
            )

            last_second_nema.offset_parts(["li", "ll", "ul"])

            for part in parts:
                last_second_nema.maya_data[part] = last_second_nema.maya_data[
                    part
                ][-10:]
                last_ema_frame[part][0] = last_second_nema.maya_data[
                    part
                ][-1:][0]

            output_frames = last_second_nema.maya_data
        else:
            if cumulative_npy.shape == (0, 12):
               cumulative_npy = first_npy_frame
            else:
                cumulative_npy = np.vstack(
                    [cumulative_npy, cumulative_npy[-10:]]
                )

            frames_data = {}
            for part in parts:
                frames_data[part] = []
                for _ in range(10):
                    frames_data[part].append(last_ema_frame[part][0])
            output_frames = frames_data
        print(cumulative_npy.shape)

        return output_frames, cumulative_audio, cumulative_npy, last_ema_frame