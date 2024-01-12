import socket
import pickle
import maya.cmds as cmds
import numpy as np
import asyncio
import shared_memory
import time

class RecvWav2Maya():
    axes = ["X", "Y", "Z"]
    #parts = ['tt', 'tb', 'td', 'li', 'll', 'ul']
    parts = ['tt', 'tb', 'td', 'li']

    def __init__(self):
        try:
            self.shm = shared_memory.SharedMemory(name="audio_stream", size=300000)
        except:
            self.shm = shared_memory.SharedMemory(
                name="audio_stream", 
                create=True,
                size=300000
                )
        self.frame_num = 0
    
    def translate(self, axis:int, mesh:str, value:float):
        cmds.setAttr(f"{mesh}.translate{self.axes[axis]}", value)

    def key_translate(self, axis:int, mesh:str, key:int, value:float):
        cmds.setKeyframe(
            mesh,
            time=key,
            attribute=f"translate{self.axes[axis]}",
            value=value,
        )

    def animate_translate(self, axis:int, mesh:str, keys):
        for key in keys:
            if type(key[axis + 1]) in [int, float]:
                value = key[axis + 1]
            elif type(key[axis + 1]) == np.float32:
                value = key[axis + 1].item()

            self.translate(axis, mesh, value)
            self.key_translate(axis, mesh, key[0], value)
            cmds.refresh()

    def print_dict(self, data):
        for part in self.parts:
            print(f"{part}: {data[part]}")

    def animate_mouth(self, maya_data, offset_li=True):
        first_part_data = maya_data[self.parts[0]]
        for i in range(len(first_part_data)):
            for part in self.parts:
                key = maya_data[part][i]
                x_value = self.get_value(key, 2)
                y_value = self.get_value(key, 1)
                mesh = f"{part}Handle"

                self.translate(2, mesh, x_value)
                self.translate(1, mesh, y_value)

                #key_translate(2, mesh, frame_num, x_value)
                #key_translate(1, mesh, frame_num, y_value)
            cmds.refresh()
            #print(frame_num)
            #part = "ll"
            #key = maya_data[part][i]
            self.frame_num += 1

    def get_value(self, key, axis):
        if type(key[axis + 1]) in [int, float]:
            value = key[axis + 1]
        elif type(key[axis + 1]) == np.float32:
            value = key[axis + 1].item()
        return value
    
    def listen_play(self):
        i = 0
        animate_times = []
        while i < 10:
            print(i)
            bdata = self.shm.buf[:].tobytes()
            if not bdata[0] == 0:
                try:
                    data = pickle.loads(self.shm.buf[:])
                    #print(f"len: {len(data)}")
                    if len(data) >= 2:
                        for elem in data:
                            print(elem)
                            if elem[0] == i:
                                animate_start = time.time()
                                self.animate_mouth(elem[1])
                                i += 1
                                animate_end = time.time()
                                animate_times.append(animate_end - animate_start)
                except:
                    pass
        self.shm.close()

if __name__ == "__main__":
    recv = RecvWav2Maya()
    recv.listen_play()