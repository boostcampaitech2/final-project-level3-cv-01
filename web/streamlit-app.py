import streamlit as st
import numpy as np
import sys
import os
import tempfile
import torch
sys.path.append(os.getcwd())
import cv2 
import time
import utils.SessionState as SessionState
from random import randint
from streamlit.server.server import Server

from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device


device = select_device('')

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

def load_yolo_model(pt_file, device):
    """
    wrapper func to load and cache object detector 
    """
    obj_detector = attempt_load(pt_file, map_location=device)

    return obj_detector

def DetermineBoxCenter(box):
    cx = int(box[0] + (box[2]/2))
    cy = int(box[1] + (box[3]/2))

    return [cx, cy]    

def drawBoxes(frame, pred, thres = 0.9): # thres 조절 추가 예정
    pred = pred.to('cpu')
    boxColor = (128, 255, 0) # very light green
    boxColor = {
        0: (128, 255, 0),
        1: (255, 255, 0),
        2: (0, 0, 255),
        3: (255, 0, 0),
    }
    className = {
        0: "Helmet",
        1: "NoHelmet",
        2: "SharingHelmet",
        3: "Sharing",
    }
    TextColor = (255, 255, 255) # white
    boxThickness = 3 
    textThickness = 2

    for x1, y1, x2, y2, conf, lbl in pred:
        if conf < thres:
            break
        lbl = int(lbl)
        if lbl not in [0,1,2,3]:
            continue
        x1, y1, x2, y2, conf = int(x1), int(y1), int(x2), int(y2), float(conf) # tensor to float
        start_coord = (x1, y1)
        # w, h = box[2:]
        # end_coord = start_coord[0] + w, start_coord[1] + h
        end_coord = (x2, y2)
        cx, cy = int(x1 + x2/2), int(y1 + y2/2) # 박스중심좌표
    # text to be included to the output image
        txt = f'{className[lbl]} ({round(conf, 3)})'
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

    model = load_yolo_model('yolor-d6.pt', device).eval()


    with upload:
        f = st.file_uploader('Upload Video file', key = state.upload_key)
    
    if f is not None:
        tfile = tempfile.NamedTemporaryFile(delete = False)
        tfile.write(f.read())  
        upload.empty()
        vf = cv2.VideoCapture(tfile.name)

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
                ProcessFrames(vf, model, stop_button)
            else:
                state.run = True
                trigger_rerun()
    vf = cv2.VideoCapture('/opt/ml/video/GOPR1296.MP4')
    ProcessFrames(vf, model, stop_button)

def ProcessFrames(vf, obj_detector, stop): 
    """
        main loop for processing video file:
        Params
        vf = VideoCapture Object
        obj_detector = Object detector (model and some properties) 
    """
    frameWidth = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))	# 영상의 넓이(가로) 프레임
    frameHeight = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
    fourcc = "mp4v"  # output video codec
    vid_writer = cv2.VideoWriter(
                            "/opt/ml/video/result.mp4", cv2.VideoWriter_fourcc(*fourcc), fps, (1280, 960)
                        )

    while vf.isOpened():
        # if frame is read correctly ret is True
        ret, frame = vf.read()
        try:
            frame = cv2.resize(frame, (1280, 960)) # 추후 조절하는 기능 추가할 예정
        except: 
            print('resize failed :', frame_counter)
            if frame_counter/num_frames == 1:
                break
            # frame_counter += 1
            continue

        if _stop:
            break
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
        frame_tensor = np.transpose(frame, (2, 0, 1))
        frame_tensor = torch.from_numpy(frame_tensor).to(device)
        frame_tensor = frame_tensor.float()  # uint8 to fp16/32
        frame_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
        if frame_tensor.ndimension() == 3:
            frame_tensor = frame_tensor.unsqueeze(0)
        pred = obj_detector(frame_tensor)[0]
        pred = non_max_suppression(pred)[0]
        frame = drawBoxes(frame, pred) 

        end = time.time()

        frame_counter += 1
        fps_measurement = frame_counter/(end - start)
        fps_meas_txt.markdown(f'**Frames per second:** {fps_measurement:.2f}')
        # import pdb; pdb.set_trace()
        print(frame_counter/num_frames)
        bar.progress(frame_counter/num_frames)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # stframe.image(frame, width = 720)
        
        vid_writer.write(frame)
    print('finish!')
    video_file = open("/opt/ml/video/result.mp4", 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
main()