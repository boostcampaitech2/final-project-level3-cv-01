import io
import os
import cv2
import numpy as np
import torch
import streamlit as st

from PIL import Image
from gcloud import storage
from oauth2client.service_account import ServiceAccountCredentials


def DetermineBoxCenter(box):
    cx = int(box[0] + (box[2]/2))
    cy = int(box[1] + (box[3]/2))

    return [cx, cy] 


def drawBoxes(frame, pred, classes=['AH','A~H','~AH','~A~H'], thres = 0.2): # thres 조절 추가 예정
    pred_list = []
    pred = pred.to('cpu')
    boxColor = {
        0: (128, 255, 0), # 헬멧O 혼자O 초록색
        1: (255, 255, 0), # 헬멧X 혼자O 하늘색
        2: (0, 0, 255), # 헬멧O 혼자X 빨간색
        3: (255, 0, 0), # 헬멧X 혼자X 파란색
    }

    clsToName = {'H':'Helmet', '~H':'No Helmet', 'A':'Alone','~A':'Sharing','AH':'Helmet','A~H':'NoHelmet','~AH':'SharingHelmet','~A~H':'SharingNoHelmet'}
    className = {}
    for i, cls in enumerate(classes):
        className[i] = clsToName[cls]

    TextColor = (255, 255, 255) # white
    boxThickness = 3 

    for x1, y1, x2, y2, conf, lbl in pred:
        if conf < thres:
            break
        lbl = int(lbl)
        if lbl not in [0,1,2,3]:
            continue
        x1, y1, x2, y2, conf = int(x1), int(y1), int(x2), int(y2), float(conf) # tensor to int or float
        start_coord = (x1, y1)
        end_coord = (x2, y2)
        pred_list.append([start_coord, end_coord, conf, lbl])
        # text to be included to the output image
        txt = f'{className[lbl]} ({round(conf, 3)})'
        frame = cv2.rectangle(frame, start_coord, end_coord, boxColor[lbl], boxThickness)
        frame = cv2.putText(frame, txt, start_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, TextColor, 2)

    return frame, pred_list
    

def lookup_checkpoint_files():

    file_list = list(os.listdir('./checkpoint'))
    file_list.sort()
    checkpoint_file_list = []
    for file in file_list:
        if file[-3:] == '.pt':
            checkpoint_file_list.append(file)
    
    checkpoint_file_list.append('merge')

    return tuple(checkpoint_file_list)


def np_to_tensor(image, device):
    
    image_tensor = np.transpose(image, (2, 0, 1))
    image_tensor = torch.from_numpy(image_tensor).to(device)
    image_tensor = image_tensor.float()  # uint8 to fp16/32
    image_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
    if image_tensor.ndimension() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def image_to_byte_array(image:Image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='png')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


def send_to_bucket(image_name, image_bytes):

    credentials_dict = {
        'type': 'service_account',
        'client_id': st.secrets['gcp']['client_id'],
        'client_email': st.secrets['gcp']['client_email'],
        'private_key_id': st.secrets['gcp']['private_key_id'],
        'private_key': st.secrets['gcp']['private_key'],
    }
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(
        credentials_dict
    )
    client = storage.Client(credentials=credentials, project=st.secrets['gcp']['project_id'])
    bucket = client.get_bucket(st.secrets['gcp']['bucket'])
    bucket.blob(image_name).upload_from_string(image_bytes)
    image_url = bucket.blob(image_name).public_url
    return image_url