from fastapi import FastAPI, File, BackgroundTasks, UploadFile
from fastapi.logger import logger
from fastapi.responses import Response

import cv2
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import non_max_suppression
from utils.torch_utils import select_device
from utils.prototype import drawBoxes, np_to_tensor

import time
import io
import os
import numpy as np
from PIL import Image

device = select_device('')
detector_model = attempt_load("./yolor-d6.pt", map_location=device)

app = FastAPI(
    title = "Detection of misconduct",
    description = "Personal Mobility detection of misconduct",
    version="0.1.0",
)

task = {}


@app.post("/detection/image/")
def post_predict_detector_image(file: bytes = File(...), width: int = 1280, height: int = 960, conf: float = 0.7, ckpt_file: str = 'yolor-d6.pt'):
    logger.info("get image")
    image = Image.open(io.BytesIO(file))
    cv_image = np.array(image)
    cv_image = cv2.resize(cv_image, (width, height))
    tensor_image = np_to_tensor(cv_image, device)
    pred = detector_model(tensor_image)[0]
    pred = non_max_suppression(pred)[0]
    converted_img = drawBoxes(cv_image, pred, conf) 
    converted_img = Image.fromarray(converted_img)
    bytes_io = io.BytesIO()
    converted_img.save(bytes_io, format="PNG")
    return Response(bytes_io.getvalue(), media_type="image/png")

   
@app.post("/detection/video/")
async def post_predict_detector_video(file: UploadFile = File(...), width: int = 1280, height: int = 960, conf: float = 0.7, ckpt_file: str = 'yolor-d6.pt'):
    logger.info(f"Post Succes Video")
    name = f"result.mp4"
    logger.info(f"file: {name}")
    cap = cv2.VideoCapture(file.filename)
    print(cap)
    try:
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) 
        print('Total number of frames to be processed:', num_frames,
        '\nFrame rate (frames per second):', fps)
    except:
        print('We cannot determine number of frames and FPS!')
    frame_counter = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # output video codec
    # Warning: 마지막 파라미터(이미지 크기 예:(1280, 960))가 안 맞으면 동영상이 저장이 안 됨!
    video_writer = cv2.VideoWriter("/opt/ml/final_project/backend/result.mp4", fourcc, fps, (width, height))
    start = time.time()
    while cap.isOpened():
        # if frame is read correctly ret is True
        ret, frame = cap.read()
        try:
            frame = cv2.resize(frame, (width, height))
        except: 
            print('resize failed :', frame_counter)
            if frame_counter/num_frames == 1:
                break
            continue
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
        frame_tensor = np_to_tensor(frame, device)
        pred = detector_model(frame_tensor)[0]
        pred = non_max_suppression(pred)[0]
        frame = drawBoxes(frame, pred, conf) 

        end = time.time()

        frame_counter += 1
        fps_measurement = frame_counter/(end - start)
        print(f'Frames per second: {fps_measurement:.2f}')
        print(frame_counter/num_frames)        
        video_writer.write(frame)
    video_writer.release()    
    print('finish!')
    

@app.get("/detection/video/status")
async def get_predict_detector_video():
    status, progress, save_path = detector_model.get_status()
    return {"status": status, "progress": progress, "save_path": save_path}