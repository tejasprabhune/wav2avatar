import maya.cmds as cmds

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


    def animateX(part, keys, offset=0):
        for key in keys:
            MayaUtils.keyXTranslate(part, key[0], key[1] + offset)


    def animateY(part, keys, offset=0):
        for key in keys:
            MayaUtils.keyYTranslate(part, key[0], key[2] + offset)


    def animateZ(part, keys, offset=0):
        for key in keys:
            MayaUtils.keyZTranslate(part, key[0], key[3] + offset)


    def animateXYZ(part, keys, offset=0):
        MayaUtils.animateX(part, keys)
        MayaUtils.animateY(part, keys)
        MayaUtils.animateZ(part, keys, offset)  