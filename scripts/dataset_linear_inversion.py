import pathlib
import wav2avatar.inversion.linear_inversion as wali
import os
import numpy as np

print(os.getcwd())
lr_model = wali.LinearInversion(ckpt="../wav2avatar/inversion/ckpts/lr_wlm_l9_mng_90_10hz.pkl", ssl_model="wavlm_large")

vctk_root = pathlib.Path("/data/all_data/VCTK/")

for f in vctk_root.rglob("*wlm_9/*.npy"):
    pred = lr_model.predict_from_file(f)

    print(f.parts)
    wlm_index = f.parts.index("wlm_9")
    all_data_index = f.parts.index("all_data")
    vctk_index = f.parts.index("VCTK")
    parts = list(f.parts)
    parts[wlm_index] = "mngu0_wlm_est"
    parts[all_data_index] = "prabhune"
    #print(parts)
    #parts.remove("all_data")
    f = pathlib.Path(*parts)
    f.parent.mkdir(parents=True, exist_ok=True)

    np.save(f, pred)