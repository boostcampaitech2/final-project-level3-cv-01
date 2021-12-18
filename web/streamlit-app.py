import streamlit as st
import numpy as np
import sys
import os
import tempfile
import torch
sys.path.append(os.getcwd())
import cv2 
from PIL import Image
import time
import utils.SessionState as SessionState
from random import randint
from streamlit.server.server import Server

from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device
from utils.prototype import drawBoxes, lookup_checkpoint_files, np_to_tensor

import pytz
import datetime as dt

device = select_device('')
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


def main():
    st.set_page_config(page_title = "PM 위법행위 감지 시스템", 
    page_icon=":scooter:")

    state = SessionState.get(upload_key = None, enabled = True, start = False, conf = 70, nms = 50, run = False)

    st.title("PM 위법행위 감지 시스템")
    st.write("영상에서 헬멧 미착용, 승차인원 초과행위를 탐지하는 시스템 입니다.")

    how_to = st.empty()
    with how_to.container():
        st.write(" ")
        st.write("- 사용법 : 왼쪽에서 옵션 설정 후 이미지 혹은 영상을 아래에 넣은 후 업로드가 완료되면 start 버튼을 누르세요.")
        st.subheader("사이드바 메뉴")
        st.write("- Checkpoint file : 체크포인트 파일을 선택합니다. 선택하신 체크포인트 파일 기반으로 yolor 모델을 불러와 inference 합니다.")
        st.write("- Confidense score threshold : bound box를 표시할 threshold입니다. 높을수록 confidence score가 높은 객체만 박스를 칩니다.")
        st.write("- result resolution : 결과 이미지 혹은 동영상의 해상도를 선택합니다.")


    upload = st.empty()
    start_button = st.empty()
    stop_button = st.empty()

    current_frame = st.empty()

    with upload:
        f = st.file_uploader('Upload Image or Video file', key = state.upload_key)

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
            ("512 x 512", "1280 x 960",)
        )

        if result_resolution == "1280 x 960":
            width, height = 1280, 960
        elif result_resolution == "512 x 512":
            width, height = 512, 512
    
    filepath = '/opt/ml/final_project/web/result.mp4'
    filepath_h264 = '/opt/ml/final_project/web/result_264.mp4'

    if f is not None:
        how_to.empty()
        st.sidebar.subheader("잡았다 요놈!")
        tfile = tempfile.NamedTemporaryFile(delete = False)
        tfile.write(f.read())  
        upload.empty()
        if f.type.split('/')[0].lower() == 'image':
            vf = Image.open(tfile.name)
            st.image(vf)
        else:
            vf = cv2.VideoCapture(tfile.name)
        print(type(vf))

        if not state.run:
            start = start_button.button("start")
            state.start = start

        if state.start:
            start_button.empty()
            state.enabled = False
            if state.run:
                tfile.close()
                f.close()
                state.upload_key = str(randint(1000, int(1e6)))
                state.enabled = True
                state.run = False
                
                if os.path.exists(filepath):
                    os.remove(filepath)
                if os.path.exists(filepath_h264):
                    os.remove(filepath_h264)

                model = attempt_load(f'/opt/ml/final_project/web/{ckpt_file}', map_location=device)

                if isinstance(vf, cv2.VideoCapture):                       
                    ProcessFrames(vf, model, stop_button, confidence_threshold, width, height, current_frame)
                else:
                    ProcessImage(vf, model, confidence_threshold, width, height)
            else:
                state.run = True
                trigger_rerun()


def ProcessImage(image_vf, obj_detector, confidence_threshold, width, height):
    image_np = np.array(image_vf) #pil to cv
    image_resize = cv2.resize(image_np, (width, height))
    img = Image.fromarray(image_resize)
    image_tensor = np_to_tensor(image_resize, device)

    pred = obj_detector(image_tensor)[0]
    pred = non_max_suppression(pred)[0]
    image, pred_list = drawBoxes(image_resize, pred, confidence_threshold)
    now = dt.datetime.now(KST).isoformat().split('.')[0]
    st.image(image)
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


def ProcessFrames(vf, obj_detector, stop, confidence_threshold, width, height, current_frame): 
    """
        main loop for processing video file:
        Params
        vf = VideoCapture Object
        obj_detector = Object detector (model and some properties) 
    """
    try:
        num_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(vf.get(cv2.CAP_PROP_FPS)) 
        print('Total number of frames to be processed:', num_frames,
        '\nFrame rate (frames per second):', fps)
    except:
        print('We cannot determine number of frames and FPS!')


    frame_counter = 0
    processing_discript = st.empty()
    processing_discript.write("👆처리중인 영상의 모습입니다.")
    _stop = stop.button("stop")
    fps_meas_txt = st.empty()
    bar = st.progress(frame_counter)
    start = time.time()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # output video codec
    video_writer = cv2.VideoWriter(
                            "/opt/ml/final_project/web/result.mp4", fourcc, fps, (width, height)
                        ) # Warning: 마지막 파라미터(이미지 크기 예:(1280, 960))가 안 맞으면 동영상이 저장이 안 됨!

    while vf.isOpened():
        # if frame is read correctly ret is True
        ret, frame = vf.read()
        try:
            frame = cv2.resize(frame, (width, height))
        except: 
            print('resize failed :', frame_counter)
            if frame_counter/num_frames == 1:
                break
            continue

        if _stop:
            break
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
        img = Image.fromarray(frame)
        frame_tensor = np_to_tensor(frame, device)
        pred = obj_detector(frame_tensor)[0]
        pred = non_max_suppression(pred)[0]
        frame, pred_list = drawBoxes(frame, pred, confidence_threshold)
        cvt_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_frame.image(cvt_frame)
        print(type(frame)) 
        end = time.time()
        now = dt.datetime.now(KST).isoformat()
        for i in pred_list:
            start_p = i[0]
            end_p = i[1]
            conf = i[2]
            label = i[3]
            crop_resion = (start_p + end_p)
            crop_img = img.crop(crop_resion)
            # crop_img = crop_img.convert("BGR")
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

        frame_counter += 1
        fps_measurement = frame_counter/(end - start)
        fps_meas_txt.markdown(f'**Frames per second:** {fps_measurement:.2f}')
        bar.progress(frame_counter/num_frames)        
        video_writer.write(frame)

    video_writer.release()    
    print('finish!')
    with st.spinner(text="Detecting Finished! Converting Video Codec..."):
        os.system("ffmpeg -i /opt/ml/final_project/web/result.mp4 -vcodec libx264 /opt/ml/final_project/web/result_h264.mp4 -y")
    video_file = open("/opt/ml/final_project/web/result_h264.mp4", 'rb')
    video_bytes = video_file.read()
    processing_discript.empty()
    current_frame.empty()
    st.video(video_bytes)


main()