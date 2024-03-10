import maya.cmds as cmds

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
    cmds.playblast(st=start, et=end, f=filename, s=sound_node, w=width, h=height, fmt="movie", p=100, os=True, orn=False)
   
playblast_avatar(start=0, end=800, width=1280, height=720, filename="C:\\Users\\tejas\\Documents\\UCBerkeley\\bci\\wav2avatar\\wav2avatar\\inversion\\ema\\pataka_hfcar_conv", sound_node="pataka")