from turtle import width
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

    resp = requests.post(
        server + f"/detection/image/?width={width}&height={height}&conf={conf}&ckpt_file={ckpt_file}",
        data=m,
        headers={"Content-Type": m.content_type},
        timeout=8000,
    )

    return resp


def detect_video(server):

    requests.post(
        server + "/detection/video",
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

    upload = st.empty()
    start_button = st.empty()
    stop_button = st.empty()


    with upload:
        data_type = st.selectbox("Choose Data Type", ["Image", "Video"])
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
            ("1280 x 960",)
        )

        if result_resolution == "1280 x 960":
            width, height = 1280, 960
        elif result_resolution == "640 x 480":
            width, height = 640, 480
    
    filepath = '/opt/ml/final_project/web/result.mp4'
    filepath_h264 = '/opt/ml/final_project/web/result_264.mp4'

    if input_data is not None:
        
        # tfile = tempfile.NamedTemporaryFile(delete = False)
        # tfile.write(f.read())  
        # upload.empty()
        # if f.type.split('/')[0].lower() == 'image':
        #     #vf = Image.open(tfile.name)
        #     #st.image(vf)
        #     files = [('files', (tfile.name, tfile.type))]
        #     response = requests.post("http://localhost:8001/detection/image", files=files)
        # else:
        #     vf = cv2.VideoCapture(tfile.name)
        # print(type(vf))

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
                input_data.close()
            else:
                state.run = True
                trigger_rerun()


def run_app():

    data_type = st.selectbox("Choose Data Type", ["Image", "Video"])
    input_data = st.file_uploader(f"insert {data_type}")  # image upload widget
    time.sleep(1)

    if st.button("Detect Plant Disease"):

        col1, col2 = st.beta_columns(2)

        if data_type == "Image":

            if input_data:
                pred = detect_image(input_data, backend)
                original_image = Image.open(input_data).convert("RGB")
                converted_image = pred.content
                converted_image = Image.open(io.BytesIO(converted_image)).convert("RGB")
                r, g, b = converted_image.split()
                converted_image = Image.merge("RGB", (b, g, r))

                col1.header("Original")
                col1.image(original_image, use_column_width=True)
                col2.header("Detected")
                col2.image(converted_image, use_column_width=True)

            else:
                # handle case with no image
                st.write("Insert an image!")

        elif data_type == "Video":

            temp_path = "/var/lib/assets"
            for t in os.listdir(temp_path):
                os.remove(temp_path + "/" + t)

            origin_video = input_data.read()

            video_path = "/var/lib/assets/video1.mp4"
            if os.path.isfile(video_path):
                os.remove(video_path)

            with open(video_path, "wb") as wfile:
                wfile.write(origin_video)
                logging.info(f"{video_path} added")

            time.sleep(1)
            wfile.close()
            detect_video(backend)

            time.sleep(1)

            status = None
            bar = st.progress(0)
            # with stqdm(total=1, st_container=st) as pbar:
            while status != "Success":
                resp = get_video_status(backend)
                resp_dict = json.loads(resp.content.decode("utf-8"))
                status = resp_dict["status"]
                if status != "Pending":
                    progress = resp_dict["progress"]
                    # pbar.update(int(progress))
                    bar.progress(int(progress))

                time.sleep(1)

            time.sleep(3)

            save_path = "/var/lib/assets/detect1.mp4"
            convert_path = "/var/lib/assets/detect2.mp4"
            os.system(f"ffmpeg -i {save_path} -vcodec libx264 {convert_path}")

            video_file = open(convert_path, "rb")
            video_bytes = video_file.read()

            col1.header("Original")
            col2.header("Detected")
            col1.video(origin_video, format="video/mp4")
            col2.video(video_bytes, format="video/mp4")


if __name__ == "__main__":
    main()