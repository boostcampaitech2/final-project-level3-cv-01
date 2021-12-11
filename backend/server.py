from fastapi import FastAPI, File, BackgroundTasks
from fastapi.logger import logger
from fastapi.responses import Response

import cv2
import asyncio
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import non_max_suppression
from utils.torch_utils import select_device
from utils.prototype import drawBoxes, lookup_checkpoint_files, np_to_tensor

import io
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

@app.post("/detection/image")
def post_predict_detector_image(file: bytes = File(...)):
    logger.info("get image")
    image = Image.open(io.BytesIO(file))
    cv_image = np.array(image)
    cv_image = cv_image[:,:,::-1].copy()
    converted_img = detector_model.detect(cv_image, image_size=416)
    converted_img = Image.fromarray(converted_img)
    bytes_io = io.BytesIO()
    converted_img.save(bytes_io, format="PNG")
    return Response(bytes_io.getvalue(), media_type="image/png")

@app.post("/detection/video")
async def post_predict_detector_video(background_task: BackgroundTasks):
    logger.info(f"Post Succes Video")
    name = f"detect.mp4"
    logger.info(f"file: {name}")
    
    video_path = "???"
    cap = cv2.VideoCapture(video_path)
    
    background_task.add_task(
        detector_model.detect, cap, image_size=416, video=True, save_path=name
    )
    
@app.get("/detection/video/status")
async def get_predict_detector_video():
    status, progress, save_path = detector_model.get_status()
    return {"status": status, "progress": progress, "save_path": save_path}