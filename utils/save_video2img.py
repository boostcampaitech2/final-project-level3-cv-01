import argparse
import cv2
import time
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from pathlib import Path
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolor-p6.pt', help='model.pt path(s)')
    #parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='baek/output', help='save results to project/name')
    #parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    '''
    사용법은 python save_video2img.py --device 0 --conf {conf} --iou {iou} --weights {pt_file}
    이런 식으로 하시면 되고, 영상은 video_cap에 위치 지정해 주시면 됩니다. 
    그리고, --project 로 parser 주셔도 되고, 아니면 default값에 직접 적어주시면 됩니다.
    그리고 아래에서 한 번 더 설명하겠습니다.
    
    주석 처리한 video_cap.set(~~~)은 적용 안되는 것 같습니다. 혹시나 실행하실때 에러 뜨신다면 주석 풀어주세요.
    
    아 그리고, 미리 default 경로에 들어갈 ~~~/output 폴더를 미리 만들어주세요!
    '''

    video_cap = cv2.VideoCapture('D://T2247_Otimizer_Backup/Yolor/test3.MP4')
    # video_cap.set(cv2.CAP_PROP_FRAME_WIDTH,512)
    # video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT,512)

    if not video_cap.isOpened():
        print('COuld not open WEBCAM')
        exit()

    print('==Start Loading==')
    time.sleep(1)
    print('==End Loading==')

    weights, view_img, save_txt, imgsz = opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_dir = Path(opt.project)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA  

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model

    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # 미리 색상 저장
    names = model.module.names if hasattr(model, 'module') else model.names
    # names => [helmet, non_helmet, two_helmet, two_non_helmet] 인데 왜, 
    # helmet은 color의 첫 번째, non_helmet은 세 번째, two_helmet은 두 번째, two_non_helmet은 네 번째인가...
    # 순서대로 helmet  two_helmet non_helmet two_non_helmet
    #          초록,       빨강,     파랑,      검정  
    colors = [[0,255,0],[255,0,0],[0,0,255],[0,0,0]]
    
    '''
    이 부분에서 path_img 경로에 직접 영상 프레임별로 input으로 들어가게 되고, 
    위에서 설정한 --project 경로에 처리한 이미지가 저장됩니다. 
    output_path_img는 아마 쓰지 않는 것으로 기억하고 있습니다.
    이게 directory내에 이미지 지워줄려고 쓰는 걸로 기억합니다.
    '''
    
    cnt = 0
    path_img = 'D://T2247_Otimizer_Backup/Yolor/baek/input/'
    output_path_img = 'D://T2247_Otimizer_Backup/Yolor/baek/output/'
    while True:
        check, frame = video_cap.read()
        
        # 이 부분은 영상에서 프레임이 None값으로 들어올 때가 있어서 체크합니다.
        if check == False:
            continue
            
        # 이 부분에서 width, heights 크기 조절 합니다.
        h,w = int(frame.shape[0]/2), int(frame.shape[1]/2)
        
        frame = cv2.resize(frame,(w,h))
        cv2.imwrite(path_img + 'video' + str(cnt).zfill(6) + '.jpg', frame)
        #detect(path_img,cnt)


        source = path_img + 'video' + str(cnt).zfill(6) + '.jpg'


        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, auto_size=64)

        # Get names and colors
        # if cnt == 0:
        #     # 이렇게 랜덤으로 지정해주는 방법과, while 문 밖에서 미리 컬러 지정해주는 방법을 둘 다 사용 가능함.
        #     names = model.module.names if hasattr(model, 'module') else model.names
        #     colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        
        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0 = Path(path), '', im0s
                #save_path = str(save_dir / p.name)
                save_path = str(save_dir / p.name)
                txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))


                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'images':
                        im0 = cv2.resize(im0,(w,h))
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fourcc = 'mp4v'  # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        vid_writer.write(im0)
                
                view_img = True
                # Stream results
                
                if view_img:
                    img_output = cv2.imread(save_path)
                    #cv2.imshow(save_path, im0)
                    cv2.imshow('output',img_output)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration
                    #cv2.destroyAllWindows()




        #os.remove(output_path_img + 'video' + str(cnt).zfill(6) + '.jpg')
        #os.remove(path_img + 'video' + str(cnt).zfill(6) + '.jpg')

        cnt += 1
        key = cv2.waitKey(1)
        if key==27:
            break
        


    video_cap.release()
    cv2.destroyAllWindows()