import streamlit as st
import numpy as np
import sys
import os
import tempfile
import torch
sys.path.append(os.getcwd())
# import traffic_counter as tc
import cv2 
import time
import utils.SessionState as SessionState
from random import randint
from streamlit import caching
import streamlit.report_thread as ReportThread 
from streamlit.server.server import Server
import copy
# from components.custom_slider import custom_slider

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from pathlib import Path


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
    # this_session.request_rerun(None)
    st.experimental_rerun()

def load_yolo_model(pt_file):
    """
    wrapper func to load and cache object detector 
    """
    device = select_device('cpu')
    obj_detector = attempt_load(pt_file, map_location=device)

    return obj_detector

def DetermineBoxCenter(box):
    cx = int(box[0] + (box[2]/2))
    cy = int(box[1] + (box[3]/2))

    return [cx, cy]    

def drawBoxes(frame, pred):
    boxColor = (128, 255, 0) # very light green
    boxColor = {
        0: (128, 255, 0),
        1: (255, 255, 0),
        2: (0, 0, 255),
        3: (255, 0, 0),
    }
    TextColor = (255, 255, 255) # white
    boxThickness = 3 
    textThickness = 2

    for x1, y1, x2, y2, conf, lbl in pred:
        start_coord = (x1, y1)
        # w, h = box[2:]
        # end_coord = start_coord[0] + w, start_coord[1] + h
        end_coord = (x2, y2)
        cx, cy = int(x1 + x2/2), int(y1 + y2/2) # 박스중심좌표
    # text to be included to the output image
        txt = '{} ({})'.format(', '.join([str(i) for i in [cx, cy]]), round(conf,3))
        frame = cv2.rectangle(frame, start_coord, end_coord, boxColor[lbl], boxThickness)
        frame = cv2.putText(frame, txt, start_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, TextColor, 2)

    return frame


def main():
    st.set_page_config(page_title = "킥보드 부정이용 탐지기", 
    page_icon=":scooter:")

    state = SessionState.get(upload_key = None, enabled = True, start = False, conf = 70, nms = 50, run = False)

    upload = st.empty()
    start_button = st.empty()
    stop_button = st.empty()

    model = load_yolo_model('yolor-d6.pt')


    with upload:
        f = st.file_uploader('Upload Video file', key = state.upload_key)
        print(f)
    
    if f is not None:
        tfile = tempfile.NamedTemporaryFile(delete = False)
        tfile.write(f.read())  
        upload.empty()
        vf = cv2.VideoCapture(tfile.name)
        frameWidth = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
        print('frameWidth', frameWidth)
        if not state.run:
            start = start_button.button("start")
            state.start = start

        if state.start:
            start_button.empty()
            #state.upload_key = str(randint(1000, int(1e6)))
            state.enabled = False
            if state.run:
                tfile.close()
                f.close()
                state.upload_key = str(randint(1000, int(1e6)))
                state.enabled = True
                state.run = False
                print(vf)
                ProcessFrames(vf, model, stop_button)
            else:
                state.run = True
                trigger_rerun()


def ProcessFrames(vf, obj_detector,stop): 
    """
        main loop for processing video file:
        Params
        vf = VideoCapture Object
        tracker = Tracker Object that was instantiated 
        obj_detector = Object detector (model and some properties) 
    """
    frameWidth = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))	# 영상의 넓이(가로) 프레임
    frameHeight = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(frameWidth, frameHeight)
    try:
        num_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(vf.get(cv2.CAP_PROP_FPS)) 
        print('Total number of frames to be processed:', num_frames,
        '\nFrame rate (frames per second):', fps)
    except:
        print('We cannot determine number of frames and FPS!')


    frame_counter = 0
    _stop = stop.button("stop")
    # new_car_count_txt = st.empty()
    fps_meas_txt = st.empty()
    bar = st.progress(frame_counter)
    stframe = st.empty()
    start = time.time()

    while vf.isOpened():
        # if frame is read correctly ret is True
        ret, frame = vf.read()
        if _stop:
            break
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
        frame = np.transpose(frame, (2, 0, 1))
        frame = torch.from_numpy(frame)
        frame = frame.float()  # uint8 to fp16/32
        frame /= 255.0  # 0 - 255 to 0.0 - 1.0
        if frame.ndimension() == 3:
            frame = frame.unsqueeze(0)
        # import pdb; pdb.set_trace()
        pred = obj_detector(frame)[0]
        pred = non_max_suppression(pred)
        print(pred)
        # (x1, y1, x2, y2, conf, cls)
        # labels, current_boxes, confidences = obj_detector.ForwardPassOutput(frame)
        frame = drawBoxes(frame, pred) 
        # new_car_count = tracker.TrackCars(current_boxes)
        # new_car_count_txt.markdown(f'**Total car count:** {new_car_count}')

        end = time.time()

        frame_counter += 1
        fps_measurement = frame_counter/(end - start)
        fps_meas_txt.markdown(f'**Frames per second:** {fps_measurement:.2f}')
        bar.progress(frame_counter/num_frames)

        frm = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frm, width = 720)

main()