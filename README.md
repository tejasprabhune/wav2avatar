# wav2avatar - Using inversion to visualize speech in real-time

Paper submitted to ICASSP: [Towards Streaming Speech-to-Avatar Synthesis](https://cmsworkshops.com/ICASSP2024/Papers/Uploads/Proposals/PaperNum/7682/20230914063600_437098_7682.pdf)

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


### Offline



### Streaming (WIP)

1. Open `wav2avatar/maya_models/stream_model.mb`
2. Open `scripts/recv_wav2maya.py` in Maya
3. Run `scripts/stream_wav2maya.py` (replacing the model location in the code - cli coming)

When you see "allocating shared memory", run
`recv_wav2maya.py` in Maya. When you see "listening", you should be able to
speak and see the corresponding animations in Maya.

Note: mayapy in Maya 2022 uses Python 3.6 which doesn't support pickling
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
`pip` library

We do a similar thing for `torchaudio` if it throws an error.