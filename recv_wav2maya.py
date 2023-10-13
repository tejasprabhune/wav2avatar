import socket
import pickle
import maya.cmds as cmds
import numpy as np
import asyncio
import shared_memory
import time

axes = ["X", "Y", "Z"]
#parts = ['tt', 'tb', 'td', 'li', 'll', 'ul']
parts = ['tt', 'tb', 'td', 'li']
frame_num = 0

def translate(axis:int, mesh:str, value:float):
    cmds.setAttr(f"{mesh}.translate{axes[axis]}", value)

def key_translate(axis:int, mesh:str, key:int, value:float):
    cmds.setKeyframe(
        mesh,
        time=key,
        attribute=f"translate{axes[axis]}",
        value=value,
    )

def animate_translate(axis:int, mesh:str, keys):
    for key in keys:
        if type(key[axis + 1]) in [int, float]:
            value = key[axis + 1]
        elif type(key[axis + 1]) == np.float32:
            value = key[axis + 1].item()
        
        translate(axis, mesh, value)
        key_translate(axis, mesh, key[0], value)
        cmds.refresh()

def print_dict(data):
    for part in parts:
        print(f"{part}: {data[part]}")

def animate_mouth(maya_data, offset_li=True):
    global frame_num
    first_part_data = maya_data[parts[0]]
    for i in range(len(first_part_data)):
        for part in parts:
            key = maya_data[part][i]
            x_value = get_value(key, 2)
            y_value = get_value(key, 1)
            mesh = f"{part}Handle"
            
            translate(2, mesh, x_value)
            translate(1, mesh, y_value)
            
            #key_translate(2, mesh, frame_num, x_value)
            #key_translate(1, mesh, frame_num, y_value)
        cmds.refresh()
        #print(frame_num)
        #part = "ll"
        key = maya_data[part][i]
        frame_num += 1

def get_value(key, axis):
    if type(key[axis + 1]) in [int, float]:
        value = key[axis + 1]
    elif type(key[axis + 1]) == np.float32:
        value = key[axis + 1].item()
    return value
    
shm = shared_memory.SharedMemory(name="audio_stream", size=300000)
#c = np.ndarray((2,), dtype=np.int64, buffer=shm.buf)
i = 0
#data = pickle.loads(shm.buf[:])
#print(data)
animate_times = []
while i < 1900:
    bdata = shm.buf[:].tobytes()
    #print(bdata[0])
    if not bdata[0] == 0:
        #print(bdata)
        try:
            
            data = pickle.loads(shm.buf[:])
            print(f"len: {len(data)}")
            if len(data) >= 2:
                for elem in data:
                    if elem[0] == i:
                        animate_start = time.time()
                        #print(data[i][0])
                        animate_mouth(elem[1])
                        i += 1
                        animate_end = time.time()
                        animate_times.append(animate_end - animate_start)
        except KeyboardInterrupt:
            sys.exit(0)
        except:
            pass
shm.close()

with open('C:/Users/tejas/Documents/UCBerkeley/bci/SpectrogramSynthesis/wav2ema/times/stream_mngu0_animate_times.txt', 'w') as f:
    f.write(str(animate_times))
"""
class EMAProtocol(asyncio.DatagramProtocol):
    def __init__(self) -> None:
        super().__init__()
        self.UDP_IP = '127.0.0.1'
        self.UDP_PORT = 5006
        self.queue = asyncio.Queue()
    
    def datagram_received(self, data: bytes, addr) -> None:
        loop = asyncio.get_event_loop()
        loop.create_task(self.handle_data(data))
    
    async def handle_data(self, data):
        data = pickle.loads(data)
        #await asyncio.sleep(5)
        print(data[0])
        animate_mouth(data[1])

async def print_queue(queue):
    try:
        print(queue.get_nowait())
    except asyncio.QueueEmpty:
        pass

async def main(ema_protocol):
    print("--- starting server ---")

    loop = asyncio.get_running_loop()

    transport, protocol = await loop.create_datagram_endpoint(
        lambda: ema_protocol, 
        local_addr=('127.0.0.1', 5006))
    
    try:
        await asyncio.sleep(30)
    finally:
        transport.close()

ema_protocol = EMAProtocol()
asyncio.run(main(ema_protocol))

UDP_IP = "127.0.0.1"
UDP_PORT = 5006

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(20)
sock.bind((UDP_IP, UDP_PORT))
while True:
    data, addr = sock.recvfrom(12000)
    data = pickle.loads(data)
    #if data == 999:
    #    break
    #print_dict(data)
    print(data[0])
    #animate_mouth(data[1])


last_ema_frame = {
    "tt": [[0, 0, -4.196, 8.963]],
    "tb": [[0, 0, -3.058, -0.644]],
    "td": [[0, 0, -5.57, -10.255]],
    "li": [[0, 0, 0.151, 19.762]]
}

animate_mouth(last_ema_frame)
cmds.setAttr("defaultRenderGlobals.currentRenderer", "mayaHardware2", type="string")
cmds.setAttr("defaultResolution.width", 640)
cmds.setAttr("defaultResolution.height", 480)
for i in range(0, 500):
    cmds.currentTime(i)
    cmds.setAttr("defaultRenderGlobals.imageFilePrefix", f"{i}", type="string")
    cmds.ogsRender(cam="camera2")"""