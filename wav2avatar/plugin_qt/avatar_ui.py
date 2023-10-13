import maya.cmds as cmds
from maya import OpenMayaUI as omui
from shiboken2 import wrapInstance
from PySide2 import QtUiTools, QtCore, QtGui, QtWidgets
from functools import partial
import sys

class AvatarUI(QtWidgets.QWidget):

    window = None

    def __init__(self, parent = None):
        pass