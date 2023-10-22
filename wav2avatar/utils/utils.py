import numpy as np

class Utils:

    def log(message):
        print(f"\n--- {message} ---")

    def interpolate_batch(prev_batch, curr_batch):
        cubic_bez = (
            lambda arr, t: ((1 - t) ** 3) * arr[0]
            + 3 * ((1 - t) ** 2) * t * arr[1]
            + 3 * (1 - t) * (t**2) * arr[2]
            + (t**3) * arr[3]
        )

        interp_batch = []
        interp_batch.append(prev_batch[-1])
        interp_batch.extend(curr_batch[:3])

        ys = [cubic_bez(interp_batch, t / 3) for t in range(0, 4)]

        interpolated_next_batch = []
        interpolated_next_batch.extend(
            ys[1:]
        )  # ignore last frame from prev batch
        interpolated_next_batch.extend(curr_batch[3:])

        return interpolated_next_batch

    def get_flattened_range(arr, start, end):
        dim = arr[:, start:end]
        dim = list(
            dim.reshape(
                len(dim),
            )
        )
        return dim

    def parts_interpolate_batch(cumulative_npy, full_curr_batch):
        full_prev_batch = cumulative_npy[-10:]
        if len(full_prev_batch) < 5:
            return
        for i in range(12):
            prev_batch = Utils.get_flattened_range(full_prev_batch, i, i + 1)
            curr_batch = Utils.get_flattened_range(full_curr_batch, i, i + 1)

            interpolated_batch = Utils.interpolate_batch(prev_batch, curr_batch)
            full_curr_batch[:, i:i + 1] = np.array(interpolated_batch).reshape((len(interpolated_batch), 1))

    def update_cumulative(cumulative, update):
        if not len(cumulative):
            cumulative = update
        else:
            cumulative = np.append(cumulative, update, axis=0)
        return cumulative

    def float_to_pcm16(audio):
        ints = (audio * 32768).astype(np.int16)
        little_endian = ints.astype('<u2')
        buf = little_endian.tobytes()
        return buf

    def is_speech(vad, audio, sr):
        speech_segments = []
        for i in range(0, len(audio), sr // 100):
            if vad.is_speech(
                Utils.float_to_pcm16(audio[i : i + sr // 100]), sr
            ):
                speech_segments.append(audio[i : i + sr // 100])
        return len(speech_segments) >= 9