from turtle import back, width
import streamlit as st

# from stqdm import stqdm

import io
from PIL import Image
import requests
import time
import tempfile
import logging
import json
import os
import sys
sys.path.append(os.getcwd())
import utils.SessionState as SessionState
from random import randint
from utils.prototype import lookup_checkpoint_files
from requests_toolbelt.multipart.encoder import MultipartEncoder
from streamlit.server.server import Server

backend = "http://localhost:8000"


@st.cache(
    hash_funcs={
        st.delta_generator.DeltaGenerator: lambda x: None,
        "_regex.Pattern": lambda x: None,
    },
    allow_output_mutation=True,
)


def trigger_rerun():
    """
    mechanism in place to force resets and update widget states
    """
    session_infos = Server.get_current()._session_info_by_id.values() 
    for session_info in session_infos:
        this_session = session_info.session
    # this_session.request_rerun()
    st.experimental_rerun()


def detect_image(data, server, width, height, conf, ckpt_file):
    m = MultipartEncoder(fields={"file": ("filename", data, "image/jpeg")})
    print(m)
    resp = requests.post(
        server + f"/detection/image/?width={width}&height={height}&conf={conf}&ckpt_file={ckpt_file}",
        data=m,
        headers={"Content-Type": m.content_type},
        timeout=8000,
    )

    return resp

    
def detect_video(data, server, width, height, conf, ckpt_file):
    m = MultipartEncoder(fields={"file": ("filename", data, "video/mp4")})
    requests.post(
        server + f"/detection/video/?width={width}&height={height}&conf={conf}&ckpt_file={ckpt_file}",
        data = m,
        headers = {"Content-Type": m.content_type},
        timeout=8000,
    )


def get_video_status(server):
    resp = requests.get(
        server + "/detection/video/status",
        timeout=8000,
    )

    return resp


def main():
    st.set_page_config(page_title = "안전모 미착용, 승차인원 초과 멈춰~!", 
    page_icon=":scooter:")

    state = SessionState.get(upload_key = None, enabled = True, start = False, conf = 70, nms = 50, run = False)
    data_type = st.selectbox("Choose Data Type", ["Image", "Video"])
    upload = st.empty()
    start_button = st.empty()
    stop_button = st.empty()


    with upload:
        input_data = st.file_uploader('Upload Image or Video file', key = state.upload_key)
        time.sleep(1)
        
        ckpt_files = lookup_checkpoint_files()

        ckpt_file = st.sidebar.radio(
            "select checkpoint file",
            ckpt_files
        )

        confidence_threshold = st.sidebar.slider("Confidence score threshold", 
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.05,
            )
        
        result_resolution = st.sidebar.radio(
            "select result video resolution",
            ("1280 x 960", "640 x 480")
        )

        if result_resolution == "1280 x 960":
            width, height = 1280, 960
            
        elif result_resolution == "640 x 480":
            width, height = 640, 480
    
    filepath = '/opt/ml/final_project/web/result.mp4'
    filepath_h264 = '/opt/ml/final_project/web/result_264.mp4'

    if input_data is not None:
        
        if not state.run:
            start = start_button.button("start")
            state.start = start

        if state.start:
            start_button.empty()
            state.enabled = False
            
            if state.run:
                state.upload_key = str(randint(1000, int(1e6)))
                state.enabled = True
                state.run = False
                
                if os.path.exists(filepath):
                    os.remove(filepath)
                if os.path.exists(filepath_h264):
                    os.remove(filepath_h264)

                if data_type == "Image":
                    resp = detect_image(input_data, backend, width, height, confidence_threshold, ckpt_file)
                    st.image(resp.content)
                    
                elif data_type == "Video":
                    detect_video(input_data, backend, width, height, confidence_threshold, ckpt_file)
                    with st.spinner(text="Detecting Finished! Converting Video Codec..."):
                        os.system("ffmpeg -i /opt/ml/final_project/backend/result.mp4 -vcodec libx264 /opt/ml/final_project/backend/result_h264.mp4")
                        video_file = open("/opt/ml/final_project/backend/result_h264.mp4", 'rb')
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                #input_data.close()
            else:
                state.run = True
                trigger_rerun()


if __name__ == "__main__":
    main()