from flask import Flask, render_template, Response, request, jsonify, stream_with_context
from aiortc import RTCPeerConnection, RTCSessionDescription

import numpy as np

import torch
import torchaudio
import s3prl.hub
import emformer
import nema_data

import cv2
import json
import uuid
import asyncio
import logging
import time
import queue

import sounddevice as sd

app = Flask(__name__, static_url_path='/static')

TEMPLATES_AUTO_RELOAD = True

async def offer_async():
    params = await request.json
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # Create an RTCPeerConnection instance
    pc = RTCPeerConnection()

    # Generate a unique ID for the RTCPeerConnection
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pc_id = pc_id[:8]

    # Create and set the local description
    await pc.createOffer(offer)
    await pc.setLocalDescription(offer)

    # Prepare the response data with local SDP and type
    response_data = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    return jsonify(response_data)

def offer():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    future = asyncio.run_coroutine_threadsafe(offer_async(), loop)
    return future.result()

def generate_frames(feature_model, emformer_model: emformer.EMAEmformer):
    q_in = queue.Queue()

    def callback(indata, frame_count, time_info, status):
        q_in.put(indata)
        #print(indata.shape)

    audio, sr = torchaudio.load("static/wav/mngu0_s1_1165.wav")
    audio_feat = feature_model(audio)
    pred = emformer_model.predict_ema(audio_feat)

    stream = sd.InputStream(samplerate=16000, callback=callback, channels=1, blocksize=1800)
    state = None
    with stream:
        print("--- Starting streaming ---")
        while True:
            indata = q_in.get()

            audio = torch.from_numpy(indata[:, 0])
            # if (emformer_model.is_speech(audio[:1600].numpy(), 16000)):
            audio = torch.reshape(audio, (1, -1))
            #print(audio.shape)
            audio_feat = feature_model(audio)
            pred, state = emformer_model.predict_ema(audio_feat, state)
            # print(state[0][1].mean())
            pred = pred.detach().cpu().numpy()
            pred = nema_data.NEMAData.mngu0_to_hprc(pred)

            fmt_pred = nema_data.NEMAData(pred, is_file=False, demean=False, normalize=False)
            # print(fmt_pred.get_json())

            yield json.dumps(fmt_pred.get_json()) + "\n"
            #time.sleep(0.01)

def generate_frames_static():
    ema = np.load("static/ema/mng_1165_emf.npy")

    time.sleep(5)
    for i in range(0, ema.shape[0], 5):
        pred = ema[i:i+5]
        fmt_pred = nema_data.NEMAData(pred, is_file=False, demean=False, normalize=False)
        print(fmt_pred.get_json()['li'])
        yield json.dumps(fmt_pred.get_json()) + "\n"
        time.sleep(0.01)
    # fmt_pred = nema_data.NEMAData(ema, is_file=False, demean=False, normalize=False)
    #print(ema.shape)
    #yield json.dumps(fmt_pred.get_json()) + "\n"

def generate_frames_audio(feature_model, emformer_model: emformer.EMAEmformer):
    audio, sr = torchaudio.load("static/wav/mngu0_s1_1165.wav")

    # time.sleep(5)
    #print("--- Starting offline ---")
    state = None
    #audio = torch.reshape(audio, (1, -1))
    #audio_feat = feature_model(audio)
    #pred, state = emformer_model.predict_ema(audio_feat, state)

    #pred = pred.detach().cpu().numpy()

    #fmt_pred = nema_data.NEMAData(pred, is_file=False, demean=False, normalize=False)
    #yield json.dumps(fmt_pred.get_json()) + "\n"

    print("--- Starting streaming ---")
    print(audio.shape)
    for i in range(0, audio.shape[1], 1600):
        curr_audio = audio[:, i:i+1600]
        audio_feat = feature_model(curr_audio)
        pred, state = emformer_model.predict_ema(audio_feat, state)
        pred = pred.detach().cpu().numpy()
        pred = nema_data.NEMAData.mngu0_to_hprc(pred)
        print(pred)

        fmt_pred = nema_data.NEMAData(pred, is_file=False, demean=False, normalize=False)
        yield json.dumps(fmt_pred.get_json()) + "\n"
        #time.sleep(0.01)



def get_feature_model():
    print("--- Getting WavLM feature extractor ---")

    feature_model = getattr(s3prl.hub, "wavlm_large")()
    feature_model = feature_model.model.feature_extractor

    print("--- Loaded WavLM feature extractor ---")

    return feature_model

def get_emformer_model():
    print("--- Getting pretrained Emformer ---")

    input_dim=512
    num_heads=16
    ffn_dim=512
    num_layers=15
    segment_length=5
    left_context_length=20

    emformer_model = emformer.EMAEmformer(
        input_dim=input_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        segment_length=segment_length,
        left_context_length=left_context_length
    )

    ckpt = torch.load(f"ckpts/emf_l{left_context_length}_r0_p{segment_length}_nh{num_heads}__nl{num_layers}_ffd{ffn_dim}_0.89.pth", map_location="cuda:0")

    emformer_model.load_state_dict(ckpt["emformer_state_dict"])

    print("--- Loaded pretrained Emformer ---")

    return emformer_model

# Route to handle the offer request
@app.route('/offer', methods=['POST'])
def offer_route():
    return offer()

@app.route('/audio_feed')
def audio_feed():
    feature_model = get_feature_model()
    emformer_model = get_emformer_model()

    return Response(generate_frames(feature_model, emformer_model))

@app.route('/audio_feed_static')
def audio_feed_static():
    return Response(generate_frames_static())

####################
#    Page routes   #
####################

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/avatar')
def avatar():
    return render_template('avatar.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')