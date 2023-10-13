# wav2avatar - Using inversion to visualize speech in real-time

Paper submitted to ICASSP: [Towards Streaming Speech-to-Avatar Synthesis](https://cmsworkshops.com/ICASSP2024/Papers/Uploads/Proposals/PaperNum/7682/20230914063600_437098_7682.pdf)

Repo cleaning in progress!

## Visualizing your own audio:

1. Download and install Maya
2. Open `maya_models/stream_model.mb`
3. Open `recv_wav2maya.py` in Maya
4. Run `stream_wav2maya.py`

When you see "allocating shared memory", run
`recv_wav2maya.py` in Maya. When you see "listening", you should be able to
speak and see the corresponding animations in Maya.

Note: mayapy in Maya 2022 uses Python 3.6 which doesn't support pickling
data protocol 4 from `multiprocessing.shared_memory`, so you should run
`mayapy -m pip install shared-memory38` before running.


## Notes on the Maya Scripting

Maya uses their own Python environment which is located as `mayapy` in 
`C:\Program Files\Autodesk\Maya<VersionNumber>\bin\` or 
`/usr/autodesk/Maya<VersionNumber>/bin/` (Linux).

Adding this to `PATH` then allows us to do `mayapy <py_file>.py`.

The second setup step needed is to install `numpy` to this separate `mayapy`
instance by running `mayapy -m pip install numpy`

The reason this is needed is to use the `maya.cmds` library, where we are able
to generate whole Maya ASCII files and polygons within those files.