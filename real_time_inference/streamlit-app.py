import streamlit as st
import numpy as np
import sys
import os
import tempfile
sys.path.append(os.getcwd())
import cv2 
from PIL import Image, ImageOps
import time
import utils.SessionState as SessionState
from random import randint
from streamlit.server.server import Server

from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device
from utils.prototype import drawBoxes, lookup_checkpoint_files, np_to_tensor


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


device = select_device('')
model = attempt_load('/opt/ml/final_project/real_time_inference/w6_side_ap50.pt', map_location=device)


def ProcessImage(image, obj_detector, confidence_threshold, width, height):
    image = np.array(image) #pil to cv
    image = cv2.resize(image, (width, height))
    
    image_tensor = np_to_tensor(image, device)

    pred = obj_detector(image_tensor)[0]
    pred = non_max_suppression(pred)[0]
    image = drawBoxes(image, pred, confidence_threshold) 
    return image


def main():

    st.set_page_config(page_title = "안전모 미착용, 승차인원 초과 멈춰~!", 
    page_icon=":scooter:")
    

    state = SessionState.get(upload_key = None, enabled = True, start = False, conf = 70, nms = 50, run = False)

    stframe = st.empty()

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
        ("512 x 512","1280 x 960", )
    )
    if result_resolution == "1280 x 960":
        width, height = 1280, 960
    elif result_resolution == "512 x 512":
        width, height = 512, 512

    while True:
        if os.path.exists('test.jpg'):
            try:
                image = Image.open('test.jpg')
                image = ImageOps.exif_transpose(image)
                stframe.image(image, width = 720)
            except OSError:
                continue
        time.sleep(0.2)



main()