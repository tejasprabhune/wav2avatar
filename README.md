# wav2avatar - Using inversion to visualize speech in real-time

Paper Pre-print: [Towards Streaming Speech-to-Avatar Synthesis](https://cmsworkshops.com/ICASSP2024/Papers/Uploads/Proposals/PaperNum/7682/20230914063600_437098_7682.pdf)

Repo cleaning in progress!

## Visualizing your own audio:

1. Download and install Autodesk Maya 2022 or 2023 (other versions not 
tested).

1. Clone this repository and `cd` into the `wav2avatar` directory:

    ```
    git clone https://github.com/tejasprabhune/wav2avatar

    cd wav2avatar
    ```

1. Install `wav2avatar` as an editable module:

    ```
    pip install -e .
    ```

1. Install `articulatory` from `https://github.com/articulatory/articulatory` and `s3prl` from `https://github.com/s3prl/s3prl` (see [s3prl notes](#s3prl--torchaudio)).

### Offline

1. Run `inversion/wav2ema.py` to generate a `.npy` file for your `.wav` audio:

    ```
    python wav2ema.py --model_dir <MODEL_DIR> --wav_name <WAV_NAME> --save_dir <SAVE_DIR>
    ```

    If you do not have a Transformer/BiGRU checkpoint for inversion, you can run `inversion/linear_inversion.py` with `ckpts/lr_wlm_l9_mng_all_10hz.pkl`
    as the checkpoint and put your `.wav` file within the `.predict` function call.

1. Open `maya_models/full_face_ema.py` in Maya 2022/2023.

1. Open `offline/animate_ema.py` in Maya by accessing `Windows/General Editors/Script Editor` then using `File/Open Script`.

1. Replace `<EMA .NPY FILE PATH>` with the path to the `.npy` file generated from inversion earlier.

1. Right click on the timeline at the bottom of Maya, select `Audio/Import Audio`, and navigate to and select the `.wav` file used for inversion. You will see the waveform show up in green over the timeline. If you set the current key to 0 then click the Play icon on the right, you will hear your audio play inside Maya.

1. Select all text in the Script Editor and press `CTRL + Enter`. This will reset all keyframes for all joints then create new keyframes for every joint according to the `.npy` inversion file.

    If you navigate to the `Outliner` and select a sample joint (e.g. `head_base`), you should see many red bars in the timeline corresponding to every newly created keyframe.

1. Set the current key to 0 and click the Play icon to the right of the timeline. You should simultaneously hear your audio and see the avatar animate.

    There is a high chance that the avatar looks very warped. This is an ongoing issue with inferring the resting position of the avatar during inversion for unseen speakers. To fix this, you may have to manually change the offset values at the bottom of
    `offline/animate_ema.py` in the Script Editor. For example, if the original file animates the `ll` in this way:

    ```
    MayaUtils.animateZ("ll", ema_handler.maya_data["ll"], 0)
    ```

    but the `ll` (lower lip) juts too far out (in the `+Z` direction), we can offset this joint such that every keyframe will shift backwards by 2 units (in the `-Z` direction):

    ```
    MayaUtils.animateZ("ll", ema_handler.maya_data["ll"], -2)
    ```

    Repeating this process for every joint until the avatar looks natural is currently the only way to achieve good offline animation (you can scrub through the timeline and verify that each modification is closer to natural speaking). This is definitely very inconvenient and we are working on avatar resting position inference (will update soon!).

### Streaming (WIP - Old SHM method)

1. Open `wav2avatar/maya_models/stream_model.mb`
2. Open `scripts/recv_wav2maya.py` in Maya
3. Run `scripts/stream_wav2maya.py` (replacing the model location in the code - cli coming)

When you see "allocating shared memory", run
`recv_wav2maya.py` in Maya. When you see "listening", you should be able to
speak and see the corresponding animations in Maya.

Note: `mayapy` in Maya 2022 uses Python 3.6 which doesn't support pickling
data protocol 4 from `multiprocessing.shared_memory`, so you should run
`mayapy -m pip install shared-memory38` before running.


## Notes on various bugs encountered so far

### Maya Scripting

Maya uses their own Python environment which is located as `mayapy` in 
`C:\Program Files\Autodesk\Maya<VersionNumber>\bin\` or 
`/usr/autodesk/Maya<VersionNumber>/bin/` (Linux).

Adding this to `PATH` then allows us to do `mayapy <py_file>.py`.

The second setup step needed is to install `numpy` to this separate `mayapy`
instance by running `mayapy -m pip install numpy`

The reason this is needed is to use the `maya.cmds` library, where we are able
to generate whole Maya ASCII files and polygons within those files.

### webrtcvad

If you run into an error installing `webrtcvad` on Windows, use

`pip install webrtcvad-wheels` or `mayapy -m pip install webrtcvad-wheels`

for the corresponding `mayapy` installation.

### s3prl + torchaudio

`s3prl` does not support Windows, but we can work around this by getting
rid of all the times `s3prl.hub` requires the `sox_io` backend after cloning
the repo, then locally installing that version instead of from the original
`pip` library.

We do a similar thing for `torchaudio` if it throws an error.