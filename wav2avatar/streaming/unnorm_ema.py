import sys
import os

sys.path.append("C:\\Users\\tejas\\Documents\\UCBerkeley\\bci\\wav2avatar\\website\\")

from importlib import reload

import nema_data
from nema_data import NEMAData
#from microphone_streaming import EMAFromMic
reload(nema_data)
from nema_data import NEMAData

import numpy as np

#inv08 = np.load(ll_path + "streamed\\mngu0_stream.npy")
#print(inv08.shape)

ll_path = "C:\\Users\\tejas\\Documents\\UCBerkeley\\bci\\language_learning\\"
wa_path = "C:\\Users\\tejas\\Documents\\UCBerkeley\\bci\\wav2avatar\\wav2avatar\\inversion\\ema\\"
study_path = "C:\\Users\\tejas\\Documents\\UCBerkeley\\bci\\wav2avatar\\wav2avatar\\inversion\\ema\\study\\mos\\"
cj_path = "C:\\Users\\tejas\\Documents\\UCBerkeley\\bci\\wav2avatar\\wav2avatar\\inversion\\ema\\cj_journal\\"
website_path = "C:\\Users\\tejas\\Documents\\UCBerkeley\\bci\\wav2avatar\\website\\static\\"
#ema_handler = NEMAData(wa_path + "mlk_pred_hb_m ng.npy", demean=True, normalize=True)
ema_handler = NEMAData(website_path + "mngu0_s1_1165_emf.npy", demean=False, normalize=False)

def clear_keys(part):
    startTime = cmds.playbackOptions(query=True, minTime=True)
    endTime = cmds.playbackOptions(query=True, maxTime=True)

    cmds.cutKey(part, time=(startTime, endTime), attribute="translateX")
    cmds.cutKey(part, time=(startTime, endTime), attribute="translateY")
    cmds.cutKey(part, time=(startTime, endTime), attribute="translateZ")


def keyYTranslate(part, key, y):
    cmds.setKeyframe(part, time=key, attribute="translateY", value=y)


def keyXTranslate(part, key, x):
    cmds.setKeyframe(part, time=key, attribute="translateX", value=x)


def keyZTranslate(part, key, z):
    cmds.setKeyframe(part, time=key, attribute="translateZ", value=z)


def animateX(part, keys):
    for key in keys:
        keyXTranslate(part, key[0], key[1])


def animateY(part, keys, offset=0, factor=1):
    for key in keys:
        keyYTranslate(part, key[0], (key[2]*factor) + offset)


def animateZ(part, keys, offset=0, factor=1):
    for key in keys:
        keyZTranslate(part, key[0], (key[3]*factor) + offset)


def animateXYZ(part, keys, offset=0):
    animateX(part, keys)
    animateY(part, keys)
    animateZ(part, keys, offset)  

print(type(ema_handler.maya_data["tt"]))

parts = ["tt", "tb", "td", "li", "ul", "ll"]
for part in parts:
    clear_keys(part)
    animateZ(part, ema_handler.maya_data[part], 5, factor=3)
    animateY(part, ema_handler.maya_data[part], -5, factor=3)
clear_keys("li_hinge")
clear_keys("upper_teeth_joint")
clear_keys("head_li")
clear_keys("head_base")
clear_keys("head_li_base")
clear_keys("tongue_base")
animateZ("ll", ema_handler.maya_data["ll"], 23, factor=3)
animateY("ll", ema_handler.maya_data["ll"], -18, factor=3)
animateZ("li_hinge", ema_handler.maya_data["li"], 0, factor=3)
animateZ("ul", ema_handler.maya_data["ul"], 28, factor=3)
animateY("ul", ema_handler.maya_data["ul"], 1, factor=2)
#animateZ("upper_teeth_joint", ema_handler.maya_data["ul"], -37)
animateZ("head_li", ema_handler.maya_data["li"], 17, factor=3)
animateY("head_li", ema_handler.maya_data["li"], -20, factor=3)
animateZ("li", ema_handler.maya_data["li"], 13, factor=3)
animateY("li", ema_handler.maya_data["li"], -15, factor=3)
#animateY("head_li", ema_handler.maya_data["li"], -7)


#animateZ("head_li_base", ema_handler.maya_data["ul"], -15, factor=0.7)
#animateZ("head_base", ema_handler.maya_data["ul"], -60)
animateZ("tongue_base", ema_handler.maya_data["li"], -20)
animateY("tongue_base", ema_handler.maya_data["li"], -10, factor=0.7)