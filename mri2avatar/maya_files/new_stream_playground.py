import maya.api.OpenMaya as om
from maya.api.OpenMayaAnim import MAnimControl as mac
import time
import maya.cmds as cmds

sel_list = om.MSelectionList()
act_sel = om.MGlobal.getActiveSelectionList()
print(act_sel.getDagPath(0))

dag_path = act_sel.getDagPath(0)

sphere_transform = om.MFnTransform(dag_path)

one_vector = om.MVector(10, 10, 10)

space = om.MSpace.kTransform

print(sphere_transform.setTranslation(one_vector, om.MSpace.kTransform))

for i in range(0, 100):
    one_vector = om.MVector(i * 2, i * 2, i * 2)
    
    sphere_transform.setTranslation(one_vector, om.MSpace.kTransform)
    
message = [10, {'li': [[98, 0, -1.9991789, -1.2260265], [99, 0, -2.1295443, -1.1221447], [100, 0, -2.9475193, -1.359417], [101, 0, -2.871951, -2.0536575], [102, 0, -3.6583958, -2.9179611], [103, 0, -3.1076326, -2.8317184], [104, 0, -3.522172, -2.7058544], [105, 0, -5.2272034, -3.0873594], [106, 0, -6.5480165, -3.5298061], [107, 0, -6.586232, -3.901474]], 'ul': [[98, 0, -1.0639572, -4.6498585], [99, 0, -1.3775697, -4.9690056], [100, 0, -1.5678377, -5.065424], [101, 0, -1.9903493, -5.031708], [102, 0, -2.31295, -5.139082], [103, 0, -2.4821718, -4.996828], [104, 0, -2.3666403, -4.7754803], [105, 0, -2.4022973, -5.171812], [106, 0, -2.3547034, -4.8943844], [107, 0, -2.17762, -5.0545826]], 'll': [[98, 0, -4.7236347, -1.5534611], [99, 0, -3.2137432, -1.5047665], [100, 0, -1.3507133, -0.96803856], [101, 0, -0.34833288, -1.2926922], [102, 0, -0.22611761, -2.0835724], [103, 0, 0.0, -1.9115944], [104, 0, -0.9772382, -1.4949951], [105, 0, -3.2615023, -2.471609], [106, 0, -4.8775425, -3.0343208], [107, 0, -7.4205093, -3.6332302]], 'tt': [[98, 0, -2.2882087, 3.4766278], [99, 0, -1.1251575, 1.9926813], [100, 0, 0.22216077, 0.43531916], [101, 0, 0.7605002, -0.023591619], [102, 0, 1.1846205, -0.8071579], [103, 0, 1.2573603, 0.15211202], [104, 0, 1.0929179, 0.8146036], [105, 0, 0.060850218, 1.2211362], [106, 0, -0.4965115, 1.8630419], [107, 0, -3.1846335, 4.758611]], 'tb': [[98, 0, -2.4390657, -6.677871], [99, 0, -1.8795847, -7.519468], [100, 0, -1.156551, -8.840692], [101, 0, -1.8421603, -9.01133], [102, 0, -2.062919, -9.537491], [103, 0, -1.7905471, -9.16675], [104, 0, -1.6343915, -8.571883], [105, 0, -1.8160565, -8.242448], [106, 0, -1.581569, -7.6843038], [107, 0, -2.6308987, -5.0879836]], 'td': [[98, 0, -6.5842986, -16.559532], [99, 0, -6.2407875, -18.10407], [100, 0, -6.683122, -19.594007], [101, 0, -6.6885214, -19.951458], [102, 0, -7.953377, -20.101608], [103, 0, -7.26202, -19.829607], [104, 0, -7.930134, -18.872952], [105, 0, -8.041533, -18.865534], [106, 0, -8.666521, -19.008429], [107, 0, -6.8170443, -16.75227]]}]

batch = message[1]
print(batch["li"])
li_info = batch["li"]
axes = ["X", "Y", "Z"]
parts = ['tt', 'tb', 'td', 'li']
def key_translate(axis:int, mesh:str, key:int, value:float):
    cmds.setKeyframe(
        mesh,
        time=key,
        attribute=f"translate{axes[axis]}",
        value=value,
    )

def clear_keys(mesh):
    cmds.cutKey(mesh, time=(0, 1000), attribute="translateX")
    cmds.cutKey(mesh, time=(0, 1000), attribute="translateY")
    cmds.cutKey(mesh, time=(0, 1000), attribute="translateZ")

def get_value(key, axis):
    if type(key[axis + 1]) in [int, float]:
        value = key[axis + 1]
    elif type(key[axis + 1]) == np.float32:
        value = key[axis + 1].item()
    return value

def animate_mouth(maya_data, last_frame):
    for part in parts:
        mesh = f"{part}Handle"
        clear_keys(mesh)
    maya_range = len(maya_data["li"])
    for i in range(maya_range):
        for part in parts:
            key = maya_data[part][i]
            x_value = get_value(key, 2)
            y_value = get_value(key, 1)
            mesh = f"{part}Handle"

            key_translate(2, mesh, i + last_frame, x_value)
            key_translate(1, mesh, i + last_frame, y_value)
    mac.setMinMaxTime(om.MTime(0), om.MTime(last_frame + maya_range))
    mac.setAnimationStartEndTime(om.MTime(0), om.MTime(last_frame + maya_range))
    mac.setCurrentTime(om.MTime(last_frame))
    mac.setPlaybackMode(0)
    mac.playForward()

animate_mouth(batch, 10)