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
import pytz
import datetime as dt

from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device
from utils.prototype import drawBoxes, lookup_checkpoint_files, np_to_tensor

# 서비스 사용자한테는 프로토타입에 있었던 체크포인트, confidence threshold, 결과 해상도 지원 X
# 대신 사이드바에 위법 사진, 위법 내용, 시간을 표시할 예정
device = select_device('')
# 체크포인트 선택!
model = attempt_load('w6_side_ap50.pt', map_location=device)
KST = pytz.timezone('Asia/Seoul')


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


def ProcessImage(image, obj_detector, confidence_threshold, width, height):
    '''
    input: image, obj_detector, confidence_threshold, width, height
    output: 박스친 이미지, label의 배열
    '''

    image_np = np.array(image) #pil to cv
    image_resize = cv2.resize(image_np, (width, height))
    img = Image.fromarray(image_resize)
    image_tensor = np_to_tensor(image_resize, device)

    pred = obj_detector(image_tensor)[0]
    pred = non_max_suppression(pred)[0]
    image, pred_list = drawBoxes(image, pred, confidence_threshold)
    now = dt.datetime.now(KST).isoformat().split('.')[0]
    for i in pred_list:
        start = i[0]
        end = i[1]
        conf = i[2]
        label = i[3]
        crop_resion = (start + end)
        crop_img = img.crop(crop_resion)
        if label == 1:
            st.sidebar.image(crop_img)
            st.sidebar.write("No Helmet")
            st.sidebar.write(f"score : {conf:.3f}")
            st.sidebar.write(f"Time : {now}")
        elif label == 2:
            st.sidebar.image(crop_img)
            st.sidebar.write("Sharing")
            st.sidebar.write(f"score : {conf:.3f}")
            st.sidebar.write(f"Time : {now}") 
        elif label == 3:
            st.sidebar.image(crop_img)
            st.sidebar.write("No Helmet & Sharing")
            st.sidebar.write(f"score : {conf:.3f}")
            st.sidebar.write(f"Time : {now}")   
    return image


def main():

    st.set_page_config(page_title = "안전모 미착용, 승차인원 초과 멈춰~!", 
    page_icon=":scooter:")
    

    state = SessionState.get(upload_key = None, enabled = True, start = False, conf = 70, nms = 50, run = False)

    stframe = st.empty()

    st.sidebar.write('hello?')

    while True:
        if os.path.exists('test.jpg'):
            try:
                image = Image.open('test.jpg')
                image = ImageOps.exif_transpose(image) # pil은 자동으로 이미지를 가로가 길도록 돌려버리는데 이를 방지하는 코드
                image = ProcessImage(image, model, 0.9, 512, 512)
                stframe.image(image, width = 720)

            except OSError:
                continue
        time.sleep(0.2)



main()