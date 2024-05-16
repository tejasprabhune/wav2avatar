import maya.cmds as cmds
from .nema_data import NEMAData
import pathlib

class MayaUtils:

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
            MayaUtils.keyXTranslate(part, key[0], key[1])


    def animateY(part, keys, offset=0, factor=1):
        for key in keys:
            MayaUtils.keyYTranslate(part, key[0], (key[2]*factor) + offset)


    def animateZ(part, keys, offset=0, factor=1):
        for key in keys:
            MayaUtils.keyZTranslate(part, key[0], (key[3]*factor) + offset)


    def animateXYZ(part, keys):
        MayaUtils.animateX(part, keys)
        MayaUtils.animateY(part, keys)
        MayaUtils.animateZ(part, keys)  
    
    def playblast_avatar(
        start=0,
        end=200,
        width=1920,
        height=1080,
        filename="",
        sound_node=""):
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
        cmds.playblast(st=start, 
                       et=end, 
                       viewer=False, 
                       f=filename, 
                       s=sound_node, 
                       w=width, 
                       h=height, 
                       fmt="qt", 
                       p=100, 
                       os=True, 
                       orn=False)
    
    def open_standalone():
        import maya.standalone
        maya.standalone.initialize(name="python")
    
    def open_file(file_path):
        cmds.file(file_path, open=True, force=True)
    
    def set_side_cam():
        cmds.setAttr("defaultColorMgtGlobals.outputTransformEnabled", True)
        for cam in cmds.ls(type='camera'):
            cmds.setAttr(cam+'.rnd', False)
        cmds.setAttr('side.rnd', True)
        cmds.displayPref(displayGradient=0)
    
    def animate_web_avatar(ema_path):
        clear_parts = ["li", "ul", "ll", "tt", "tb", "td", "li_hinge", "upper_teeth_joint", "head_li", "head_base", "head_li_base", "tongue_base"]
        for part in clear_parts:
            MayaUtils.clear_keys(part)
        
        ema_handler = NEMAData(ema_path, demean=False, normalize=False)

        MayaUtils.animateZ("tt", ema_handler.maya_data["tt"], -10, factor=7)
        MayaUtils.animateY("tt", ema_handler.maya_data["tt"], 7.5, factor=12)

        MayaUtils.animateZ("tb", ema_handler.maya_data["tb"], -20, factor=5)
        MayaUtils.animateY("tb", ema_handler.maya_data["tb"], 4, factor=10)

        MayaUtils.animateZ("td", ema_handler.maya_data["td"], -22, factor=6)
        MayaUtils.animateY("td", ema_handler.maya_data["td"], 4, factor=6)

        MayaUtils.animateZ("ll", ema_handler.maya_data["ll"], -22, factor=2)
        MayaUtils.animateY("ll", ema_handler.maya_data["ll"], 17, factor=12)

        MayaUtils.animateZ("ul", ema_handler.maya_data["ul"], -18, factor=1)
        MayaUtils.animateY("ul", ema_handler.maya_data["ul"], -2, factor=10)

        MayaUtils.animateZ("li", ema_handler.maya_data["li"], -30, factor=1)
        MayaUtils.animateY("li", ema_handler.maya_data["li"], -15, factor=10)
        
        MayaUtils.animateZ("li_hinge", ema_handler.maya_data["li"], -50, factor=1)
        
        MayaUtils.animateZ("head_li", ema_handler.maya_data["li"], -20, factor=3)
        MayaUtils.animateY("head_li", ema_handler.maya_data["li"], 5, factor=10)

        MayaUtils.animateZ("tongue_base", ema_handler.maya_data["li"], -43)
        MayaUtils.animateY("tongue_base", ema_handler.maya_data["li"], -8, factor=6)
    
    def import_wav(wav_path: pathlib.Path):
        print(wav_path)
        return cmds.sound(file=wav_path, name=wav_path.stem)