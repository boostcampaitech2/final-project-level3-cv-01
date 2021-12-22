import streamlit as st
import numpy as np
import sys
import os
import tempfile
import torch
import uuid

sys.path.append(os.getcwd())

import cv2 
from PIL import Image
import time
import utils.SessionState as SessionState
from random import randint
from streamlit.server.server import Server

from models.experimental import attempt_load
from utils.general import non_max_suppression, merge_pred
from utils.torch_utils import select_device
from utils.prototype import drawBoxes, lookup_checkpoint_files, np_to_tensor, send_to_bucket, image_to_byte_array
# from db import insert_data

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
    st.experimental_rerun()


def main():
    st.set_page_config(page_title = "PM 위법행위 감지 시스템", 
    page_icon=":scooter:")

    state = SessionState.get(upload_key = None, enabled = True, start = False, conf = 70, nms = 50, run = False)
    st.title("PM 위법행위 감지 시스템")
    awesomeImage = st.empty()
    awesomeImage.image('title_image.png')
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
            list(filter(lambda x: "helmet" not in x and "alone" not in x, ckpt_files))
        )

        if ckpt_file == 'merge':
            helmet_ckpt_files = tuple(filter(lambda x: "helmet" in x, ckpt_files))
            helmet_ckpt_file = st.sidebar.radio(
                "select helmet checkpoint file for merging",
                helmet_ckpt_files
                )
            alone_ckpt_files = tuple(filter(lambda x: "alone" in x, ckpt_files))
            alone_ckpt_file = st.sidebar.radio(
                "select alone checkpoint file for merging",
                alone_ckpt_files
                )
        
        confidence_threshold = st.sidebar.slider("Confidence score threshold", 
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.05,
            )
        
        result_resolution = st.sidebar.radio(
            "select result video resolution",
            ("512 x 320", "512 x 512", "768 x 576", "1280 x 960",)
        )

        resolution = result_resolution.split(" ")
        width = int(resolution[0])
        height = int(resolution[2])
    
    filepath = 'result.mp4'
    filepath_h264 = 'result_264.mp4'

    if f is not None:
        how_to.empty()
        awesomeImage.empty()
        st.sidebar.subheader("잡았다 요놈!")
        tfile = tempfile.NamedTemporaryFile(delete = False)
        tfile.write(f.read())  
        upload.empty()
        if f.type.split('/')[0].lower() == 'image':
            vf = Image.open(tfile.name)
            st.image(vf)
        else:
            vf = cv2.VideoCapture(tfile.name)

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
                
                if ckpt_file == 'merge':
                    model_helmet = attempt_load(f'checkpoint/{helmet_ckpt_file}',map_location=device)
                    model_alone = attempt_load(f'checkpoint/{alone_ckpt_file}',map_location=device)
                    
                    if isinstance(vf, cv2.VideoCapture):                       
                        ProcessFramesMerge(vf, model_helmet, model_alone, stop_button, confidence_threshold, width, height, current_frame)
                    else:
                        ProcessImageMerge(vf, model_helmet, model_alone, confidence_threshold, width, height)

                else:
                    model = attempt_load(f'checkpoint/{ckpt_file}', map_location=device)

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
    classes = obj_detector.module.names if hasattr(obj_detector, 'module') else obj_detector.names
    image, pred_list = drawBoxes(image_resize, pred, classes, confidence_threshold)
    now = dt.datetime.now(KST).isoformat().split('.')[0]
    st.image(image)
    for i in pred_list:
        start_p, end_p, conf, label = i
        crop_region = (start_p + end_p)
        crop_img = img.crop(crop_region)
        # crop_img_byte = image_to_byte_array(crop_img) # image to byte
        # img_name = now + "_" + str(label) + uuid.uuid4().hex + "png" # DB image name
        if label == 1:
            # img_url = send_to_bucket(img_name, crop_img_byte) # send to storage
            # insert_data(now, img_url, str(label)) # insert to DB
            st.sidebar.image(crop_img)
            if classes[label] == '~A':
                st.sidebar.write("Sharing")
            else:
                st.sidebar.write("No Helmet")
            st.sidebar.write(f"score : {conf:.3f}")
            st.sidebar.write(f"Time : {now}")
        elif label == 2:
            # img_url = send_to_bucket(img_name, crop_img_byte) # send to storage
            # insert_data(now, img_url, str(label)) # insert to DB
            st.sidebar.image(crop_img)
            st.sidebar.write("Sharing")
            st.sidebar.write(f"score : {conf:.3f}")
            st.sidebar.write(f"Time : {now}") 
        elif label == 3:
            # img_url = send_to_bucket(img_name, crop_img_byte) # send to storage
            # insert_data(now, img_url, str(label)) # insert to DB
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
    processing_txt = st.empty()
    processing_txt.write("👆처리중인 영상의 모습입니다.")
    _stop = stop.button("stop")
    fps_mean_txt = st.empty()
    bar = st.progress(frame_counter)
    start = time.time()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # output video codec
    video_writer = cv2.VideoWriter(
                            "result.mp4", fourcc, fps, (width, height)
                        ) # Warning: 마지막 파라미터(이미지 크기 예:(1280, 960))가 안 맞으면 동영상이 저장이 안 됨!

    current_catch_img = st.sidebar.empty() 
    current_catch_txt = st.sidebar.empty()

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
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_tensor = np_to_tensor(frame, device)
        pred = obj_detector(frame_tensor)[0]
        pred = non_max_suppression(pred)[0]
        classes = obj_detector.module.names if hasattr(obj_detector, 'module') else obj_detector.names
        frame, pred_list = drawBoxes(frame, pred, classes, confidence_threshold)
        cvt_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_frame.image(cvt_frame)
        end = time.time()
        now = dt.datetime.now(KST).isoformat()
        for i in pred_list:
            start_p, end_p, conf, label = i
            crop_region = (start_p + end_p)
            crop_img = img.crop(crop_region)
            # crop_img_byte = image_to_byte_array(crop_img) # image to byte
            # img_name = now + "_" + str(label) + uuid.uuid4().hex + "png" # DB image name
            if label == 1:
                # img_url = send_to_bucket(img_name, crop_img_byte) # send to storage
                # insert_data(now, img_url, str(label)) # insert to DB
                current_catch_img.image(crop_img, use_column_width='always')
                current_catch_txt.write(f"No Helmet \n score : {conf:.3f} \n Time : {now}")
            elif label == 2:
                # img_url = send_to_bucket(img_name, crop_img_byte) # send to storage
                current_catch_img.image(crop_img, use_column_width='always')
                current_catch_txt.write(f"Sharing \n score : {conf:.3f} \n Time : {now}")
                # insert_data(now, img_url, str(label)) # insert to DB
            elif label == 3:
                current_catch_img.image(crop_img, use_column_width='always')
                current_catch_txt.write(f"No Helmet & Sharing \n score : {conf:.3f} \n Time : {now}")
                # img_url = send_to_bucket(img_name, crop_img_byte) # send to storage
                # insert_data(now, img_url, str(label)) # insert to DB

        frame_counter += 1
        fps_measurement = frame_counter/(end - start)
        fps_mean_txt.markdown(f'**Frames per second:** {fps_measurement:.2f}')
        bar.progress(frame_counter/num_frames)        
        video_writer.write(frame)

    video_writer.release()    
    print('finish!')
    with st.spinner(text="Detecting Finished! Converting Video Codec..."):
        os.system("ffmpeg -i result.mp4 -vcodec libx264 result_h264.mp4 -y")
    video_file = open("result_h264.mp4", 'rb')
    video_bytes = video_file.read()
    processing_txt.empty()
    current_frame.empty()
    st.video(video_bytes)
    st.write("되돌아가시려면 사이드바 메뉴에서 아무거나 선택하세요.")


def ProcessImageMerge(image_vf, helmet_detector, alone_detector, confidence_threshold, width, height):
    image_np = np.array(image_vf) #pil to cv
    image_resize = cv2.resize(image_np, (width, height))
    img = Image.fromarray(image_resize)
    image_tensor = np_to_tensor(image_resize, device)

    pred_helmet = helmet_detector(image_tensor)[0]
    pred_alone = alone_detector(image_tensor)[0]

    pred_helmet = non_max_suppression(pred_helmet)[0]
    pred_alone = non_max_suppression(pred_alone)[0]

    pred = merge_pred(pred_helmet, pred_alone, merge_thres=0.5)
    classes = ['AH','A~H','~AH','~A~H']
    image, pred_list = drawBoxes(image_resize, pred, classes, confidence_threshold)
    now = dt.datetime.now(KST).isoformat().split('.')[0]
    st.image(image)
    for i in pred_list:
        start_p, end_p, conf, label = i
        crop_region = (start_p + end_p)
        crop_img = img.crop(crop_region)
        # crop_img_byte = image_to_byte_array(crop_img) # image to byte
        # img_name = now + "_" + str(label) + uuid.uuid4().hex + "png" # DB image name
        if label == 1:
            # img_url = send_to_bucket(img_name, crop_img_byte) # send to storage
            # insert_data(now, img_url, str(label)) # insert to DB
            st.sidebar.image(crop_img)
            st.sidebar.write("No Helmet")
            st.sidebar.write(f"score : {conf:.3f}")
            st.sidebar.write(f"Time : {now}")
        elif label == 2:
            # img_url = send_to_bucket(img_name, crop_img_byte) # send to storage
            # insert_data(now, img_url, str(label)) # insert to DB
            st.sidebar.image(crop_img)
            st.sidebar.write("Sharing")
            st.sidebar.write(f"score : {conf:.3f}")
            st.sidebar.write(f"Time : {now}") 
        elif label == 3:
            # img_url = send_to_bucket(img_name, crop_img_byte) # send to storage
            # insert_data(now, img_url, str(label)) # insert to DB
            st.sidebar.image(crop_img)
            st.sidebar.write("No Helmet & Sharing")
            st.sidebar.write(f"score : {conf:.3f}")
            st.sidebar.write(f"Time : {now}")      


def ProcessFramesMerge(vf, helmet_detector, alone_detector, stop, confidence_threshold, width, height, current_frame): 
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

    current_catch_img = st.sidebar.empty() 
    current_catch_txt = st.sidebar.empty()
    frame_counter = 0
    processing_txt = st.empty()
    processing_txt.write("👆처리중인 영상의 모습입니다.")
    _stop = stop.button("stop")
    fps_mean_txt = st.empty()
    bar = st.progress(frame_counter)
    start = time.time()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # output video codec
    video_writer = cv2.VideoWriter(
                            "result.mp4", fourcc, fps, (width, height)
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
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_tensor = np_to_tensor(frame, device)

        pred_helmet = helmet_detector(frame_tensor)[0]
        pred_alone = alone_detector(frame_tensor)[0]

        pred_helmet = non_max_suppression(pred_helmet)[0]
        pred_alone = non_max_suppression(pred_alone)[0]
        
        pred = merge_pred(pred_helmet, pred_alone, merge_thres=0.5)
        classes = ['AH','A~H','~AH','~A~H']
        frame, pred_list = drawBoxes(frame, pred, classes, confidence_threshold)
        cvt_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_frame.image(cvt_frame)
        end = time.time()
        now = dt.datetime.now(KST).isoformat()
        for i in pred_list:
            start_p, end_p, conf, label = i
            crop_region = (start_p + end_p)
            crop_img = img.crop(crop_region)
            # crop_img_byte = image_to_byte_array(crop_img) # image to byte
            # img_name = now + "_" + str(label) + uuid.uuid4().hex + "png" # DB image name
            if label == 1:
                # img_url = send_to_bucket(img_name, crop_img_byte) # send to storage
                # insert_data(now, img_url, str(label)) # insert to DB
                current_catch_img.image(crop_img, use_column_width='always')
                current_catch_txt.write(f"No Helmet \n score : {conf:.3f} \n Time : {now}")
            elif label == 2:
                # img_url = send_to_bucket(img_name, crop_img_byte) # send to storage
                current_catch_img.image(crop_img, use_column_width='always')
                current_catch_txt.write(f"Sharing \n score : {conf:.3f} \n Time : {now}")
                # insert_data(now, img_url, str(label)) # insert to DB
            elif label == 3:
                current_catch_img.image(crop_img, use_column_width='always')
                current_catch_txt.write(f"No Helmet & Sharing \n score : {conf:.3f} \n Time : {now}")
                # img_url = send_to_bucket(img_name, crop_img_byte) # send to storage
                # insert_data(now, img_url, str(label)) # insert to DB

        frame_counter += 1
        fps_measurement = frame_counter/(end - start)
        fps_mean_txt.markdown(f'**Frames per second:** {fps_measurement:.2f}')
        bar.progress(frame_counter/num_frames)        
        video_writer.write(frame)

    video_writer.release()    
    print('finish!')
    with st.spinner(text="Detecting Finished! Converting Video Codec..."):
        os.system("ffmpeg -i result.mp4 -vcodec libx264 result_h264.mp4 -y")
    video_file = open("result_h264.mp4", 'rb')
    video_bytes = video_file.read()
    processing_txt.empty()
    current_frame.empty()
    st.video(video_bytes)
    st.write("되돌아가시려면 사이드바 메뉴에서 아무거나 선택하세요.")

main()