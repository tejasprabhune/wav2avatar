import sys
sys.path.append("C:\\Users\\tejas\\Documents\\UCBerkeley\\bci\\wav2avatar\\wav2avatar\\utils\\")
from nema_data import NEMAData
from maya_utils import MayaUtils

ll_path = "C:\\Users\\tejas\\Documents\\UCBerkeley\\bci\\language_learning\\"
wa_path = "C:\\Users\\tejas\\Documents\\UCBerkeley\\bci\\wav2avatar\\wav2avatar\\inversion\\ema\\"

ema_handler = NEMAData(wa_path + "mng_mlk_pred.npy", demean=True, normalize=True)
parts = ["tt", "tb", "td", "li", "ul", "ll"]

for part in parts:
    MayaUtils.clear_keys(part)
    MayaUtils.animateXYZ(part, ema_handler.maya_data[part])

MayaUtils.clear_keys("li_hinge")
MayaUtils.clear_keys("upper_teeth_joint")
MayaUtils.clear_keys("head_li")
MayaUtils.clear_keys("head_base")
MayaUtils.clear_keys("head_li_base")
MayaUtils.clear_keys("tongue_base")

# Manual shifting of joints
MayaUtils.animateZ("ll", ema_handler.maya_data["ll"], 2)
MayaUtils.animateZ("li_hinge", ema_handler.maya_data["li"], -33)
MayaUtils.animateZ("ul", ema_handler.maya_data["ul"], 3)
MayaUtils.animateY("ul", ema_handler.maya_data["ul"], -2)
MayaUtils.animateZ("upper_teeth_joint", ema_handler.maya_data["ul"], -34)
MayaUtils.animateXYZ("head_li", ema_handler.maya_data["li"], -3)
MayaUtils.animateZ("li", ema_handler.maya_data["li"], 3)
#MayaUtils.animateY("head_li", ema_handler.maya_data["li"], -7)
MayaUtils.animateY("ll", ema_handler.maya_data["ll"], -4)

MayaUtils.animateZ("head_li_base", ema_handler.maya_data["ul"], -57)
MayaUtils.animateZ("head_base", ema_handler.maya_data["ul"], -60)
MayaUtils.animateZ("tongue_base", ema_handler.maya_data["li"], -22)