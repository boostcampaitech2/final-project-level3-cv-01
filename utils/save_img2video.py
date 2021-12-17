import cv2
import os

'''
pathIn이 결과 처리한 output 위치입니다. 현재는 로컬 windows에서 사용중이라, linux 서버에서 하실 경우 맞게 위치 바꿔주세요.
pathOut은 만들 영상의 파일명을 경로와 같이 적어주세요.
그러면 영상은 output 결과가 있는 위치에 생성됩니다.
'''

# TODO - 경로 수정
pathIn = os.path.join("D:", os.sep,'T2247_Otimizer_Backup','Yolor','baek','test3_beak_w6_side_ap50')
pathOut = os.path.join("D:", os.sep,'T2247_Otimizer_Backup','Yolor','baek','test3_beak_w6_side_ap50','test3_beak_w6_side_ap50_conf0.65_iou0.2.mp4')

fname = os.listdir(pathIn)
fname.sort()

# 영상 fps 조절하는 변수입니다.
fps = 30
frame_array = []

for idx, img_ in enumerate(fname):
    # 서버에서는 cv2.imread(pathIn + '/' + img_) 해주시면 됩니다. windows 로컬에서는 '\\' 이더라구요.
    img = cv2.imread(pathIn + '/' + img_)
    
    # 처리한 영상 크게 조절 가능합니다.
    h,w = int(img.shape[0]), int(img.shape[1])
    frame_array.append(img)

output = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, (w,h))
for i in range(len(frame_array)):
    output.write(frame_array[i])

output.release()


