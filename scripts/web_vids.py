import sys
import os
import pathlib
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

def animateY_par(part, keys, parkeys, offset=0, factor=1):
    for i in range(len(keys)):
        key = keys[i]
        parkey = parkeys[i]
        y = key[2] + parkey[2]
        keyYTranslate(part, key[0], y*factor + offset)

def animateZ_par(part, keys, parkeys, offset=0, factor=1):
    for i in range(len(keys)):
        key = keys[i]
        parkey = parkeys[i]
        z = key[3] + parkey[3]
        keyZTranslate(part, key[0], z*factor + offset)

def animateXYZ(part, keys, offset=0):
    animateX(part, keys)
    animateY(part, keys)
    animateZ(part, keys, offset)

def playblast_avatar(
    start=0,
    end=200,
    width=1920,
    height=1080,
    filename="",
    sound_node="rumble"):
    """
    Converts keyframed animation to video.
    
    Args:
        start (int): Start frame of video
        end (int): End frame of video
        width (int): Resolution width in pixels
        height (int): Resolution height in pixels
        filename (str): absolute path to output video file without extension
                  (e.g. /home/.../avatar)
        sound_node (str): name of audio node to use in video (can be found by
                    right clicking on timeline -> Audios)
    """
    cmds.playblast(st=start, et=end, viewer=False, f=filename, s=sound_node, w=width, h=height, fmt="qt", p=100, os=True, orn=False)

def generate_vid(path, pb=True):

	stem = path.stem

	ema_handler = NEMAData(path, demean=False, normalize=False)

	parts = ["li", "ul", "ll", "tt", "tb", "td"]
	for part in parts:
	    clear_keys(part)
	    #animateZ(part, ema_handler.maya_data[part], 5, factor=1)
	    #animateY(part, ema_handler.maya_data[part], 17, factor=1)

	clear_keys("li_hinge")
	clear_keys("upper_teeth_joint")
	clear_keys("head_li")
	clear_keys("head_base")
	clear_keys("head_li_base")
	clear_keys("tongue_base")

	animateZ("tt", ema_handler.maya_data["tt"], -10, factor=7)
	animateY("tt", ema_handler.maya_data["tt"], 7.5, factor=12)

	animateZ("tb", ema_handler.maya_data["tb"], -20, factor=5)
	animateY("tb", ema_handler.maya_data["tb"], 4, factor=10)

	animateZ("td", ema_handler.maya_data["td"], -22, factor=6)
	animateY("td", ema_handler.maya_data["td"], 4, factor=6)

	animateZ("ll", ema_handler.maya_data["ll"], -22, factor=2)
	animateY("ll", ema_handler.maya_data["ll"], 17, factor=12)
	
	animateZ("li_hinge", ema_handler.maya_data["li"], -50, factor=1)
	
	animateZ("ul", ema_handler.maya_data["ul"], -18, factor=1)
	animateY("ul", ema_handler.maya_data["ul"], -2, factor=10)
	
	#animateZ("upper_teeth_joint", ema_handler.maya_data["ul"], -37)
	animateZ("head_li", ema_handler.maya_data["li"], -20, factor=3)
	animateY("head_li", ema_handler.maya_data["li"], 5, factor=10)
	animateZ("li", ema_handler.maya_data["li"], -30, factor=1)
	animateY("li", ema_handler.maya_data["li"], -15, factor=10)
	#animateY("head_li", ema_handler.maya_data["li"], -7) 


	#animateZ("head_li_base", ema_handler.maya_data["ul"], -15, factor=0.7)
	#animateZ("head_base", ema_handler.maya_data["ul"], -60)
	animateZ("tongue_base", ema_handler.maya_data["li"], -43)
	animateY("tongue_base", ema_handler.maya_data["li"], -8, factor=6)
	if pb:
	    playblast_avatar(start=0, end=40, width=1080, height=1080, filename=f"C:\\Users\\tejas\\Documents\\UCBerkeley\\bci\\wav2avatar\\wav2avatar\\inversion\\ema\\cj_journal\\web\\{stem}", sound_node=stem)

web_path = pathlib.Path("C:\\Users\\tejas\\Documents\\UCBerkeley\\bci\\wav2avatar\\wav2avatar\\inversion\\ema\\cj_journal\\web\\")
pb = False
#for path in web_path.rglob("*.npy"):
#    print(path)
#    generate_vid(path, pb=pb)

generate_vid(web_path / "venture.npy", pb=pb)
#generate_vid(web_path / "spain.npy", pb=pb)
#generate_vid(web_path / "venture.npy", pb=pb)