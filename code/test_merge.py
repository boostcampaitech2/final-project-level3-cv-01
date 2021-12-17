import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, box_iou, \
    non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, clip_coords, set_logging, increment_path, remove_overlap, merge_class
from utils.metrics import ap_per_class
from utils.plots import plot_images, output_to_target
from utils.torch_utils import select_device, time_synchronized

import wandb

def test_merge(data,
         aug,
         weights_helmet=None,
         weights_alone=None,
         batch_size=32,
         imgsz=512,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         merge_thres=0.6, # for mergeing two outputs
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model_helmet=None,
         model_alone=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_conf=False,
         plots=True,
         log_imgs=0,
         agnostic=False,
         entity='perforated_line'):  # number of logged images

    set_logging()
    device = select_device(opt.device, batch_size=batch_size)
    save_txt = opt.save_txt  # save *.txt labels

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model_helmet = attempt_load(weights_helmet, map_location=device) # load FP32 model
    model_alone = attempt_load(weights_alone, map_location=device)

    imgsz = check_img_size(imgsz, s=model_helmet.stride.max())  # check img_size
    imgsz = check_img_size(imgsz, s=model_alone.stride.max())

    # Half``
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model_helmet.half()
        model_alone.half()

    # Configure
    model_helmet.eval()
    model_alone.eval()

    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    check_dataset(data)

    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs, wandb = min(log_imgs, 100), None  # ceil
    try:
        import wandb  # Weights & Biases
    except ImportError:
        log_imgs = 0

    # Dataloader
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model_helmet(img.half() if half else img) if device.type != 'cpu' else None  # run once
    _ = model_alone(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images       
    dataloader = create_dataloader(path, imgsz, batch_size, model_helmet.stride.max(), opt, pad=0.5, rect=True)[0]

    seen = 0
    #names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    names = data['names']

    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model - [32,11475,7] (x,y,w,h,obj_conf,cls_conf for each class)
            t = time_synchronized()
            inf_out_helmet, _ = model_helmet(img, augment=augment)  # inference and training outputs
            inf_out_alone, _ = model_alone(img, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

             # Run NMS - [32,-,6] (x1,y1,x2,y2,conf,cls)
            t = time_synchronized()
            output_helmet = non_max_suppression(inf_out_helmet, conf_thres=conf_thres, iou_thres=iou_thres, merge=False, classes=None, agnostic=agnostic)
            output_alone = non_max_suppression(inf_out_alone, conf_thres=conf_thres, iou_thres=iou_thres, merge=False, classes=None, agnostic=agnostic)
            t1 += time_synchronized() - t

        output = []
        # Statistics per image
        for si in range(nb):
            labels = targets[targets[:, 0] == si, 1:] # (cls,x,y,w,h)
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1


            # merge pred_helmet and pred_alone to pred

            # prediction for si th image (x1,y1,x2,y2,conf,cls)
            pred_helmet = output_helmet[si]
            pred_alone = output_alone[si]

            # if there is no box detected in pred_helmet or pred_alone
            if len(pred_helmet) == 0 or len(pred_alone) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
            
            iou = box_iou(pred_helmet[:,:4],pred_alone[:,:4])
            #iou = remove_overlap(iou)
            i,j = (iou > merge_thres).nonzero(as_tuple=False).T  

            pred_helmet = pred_helmet[i]
            pred_alone = pred_alone[j]

            nbox = len(pred_helmet)
            pred = torch.zeros_like(pred_helmet)
            
            output.append(pred)

            for box_i in range(nbox):
                # todo - box weighted merge
                if pred_helmet[box_i][4] > pred_alone[box_i][4]:
                    pred[box_i][:4] = pred_helmet[box_i][:4]
                else:
                    pred[box_i][:4] = pred_alone[box_i][:4]
                pred[box_i][4] = pred_helmet[box_i][4] * pred_alone[box_i][4]
                pred[box_i][5] = merge_class(pred_helmet[box_i][5], pred_alone[box_i][5])


            if torch.count_nonzero(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            path = Path(paths[si])
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                x = pred.clone()
                x[:, :4] = scale_coords(img[si].shape[1:], x[:, :4], shapes[si][0], shapes[si][1])  # to original
                for *xyxy, conf, cls in x:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging
            if plots and len(wandb_images) < log_imgs:
                box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                             "class_id": int(cls),
                             "box_caption": "%s %.3f" % (names[cls], conf),
                             "scores": {"class_score": conf},
                             "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                boxes = {"predictions": {"box_data": box_data, "class_labels": names}}
                wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = pred[:, :4].clone()  # xyxy
                scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break 

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
            
        # Plot images
        if plots and batch_i < 10:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # filename
            plot_images(img, targets, paths, f, names)  # labels
            f = save_dir / f'test_batch{batch_i}_pred.jpg'
            plot_images(img, output_to_target(output, width, height), paths, f, names)  # predictions
            
            f = save_dir / f'test_batch{batch_i}_pred-helmet.jpg'
            names_helmet = ['H','~H']
            plot_images(img, output_to_target(output_helmet, width, height), paths, f, names_helmet)  # predictions
            f = save_dir / f'test_batch{batch_i}_pred-alone.jpg'
            names_alone = ['A','~A']
            plot_images(img, output_to_target(output_alone, width, height), paths, f, names_alone)  # predictions

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, fname=save_dir / 'precision-recall_curve.png')
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)


    # wandb project, entity 설정 해주세요!!!
    wandb_run = wandb.init(config=opt, resume="allow",
                        project='final_yolor' if opt.project == 'runs/test' else Path(opt.project).stem,
                        entity=entity,
                        name=save_dir.stem)

    # W&B logging
    if plots and wandb:
        print('---')
        wandb.log({"metrics/precision":mp})
        wandb.log({"metrics/recall":mr})
        wandb.log({"metrics/map50":map50})
        wandb.log({"metrics/map":map})
        wandb.log({"Images": wandb_images})
        wandb.log({"Validation": [wandb.Image(str(x), caption=x.name) for x in sorted(save_dir.glob('test*.jpg'))]})

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = glob.glob('../coco/annotations/instances_val*.json')[0]  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stzats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print('ERROR: pycocotools unable to run: %s' % e)

    # Return results
    print('Results saved to %s' % save_dir)
    model_helmet.float()  # for training
    model_alone.float()
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')

    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--entity', type=str, default='perforated_line', help='wandb entity name')

    parser.add_argument('--weights_helmet', nargs='+', type=str, default='runs/train/helmet/weights/best_ap.pt', help='model.pt path(s)')
    parser.add_argument('--weights_alone', nargs='+', type=str, default='runs/train/alone/weights/best_ap50.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/both.yaml', help='*.data path')

    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--merge-thres',type=float, default=0.5,help='IOU threshold for merging outputs from helmet and alone')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")

    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # test.py에서 img augmentation을 할 것인지 안 할것인지, 디폴트 값은 하는 것으로
    parser.add_argument('--aug', type=str, default='n', help='in test py img augmentation')
    parser.add_argument('--agnostic', action='store_true', help='merge boxes with different classes during nms')


    opt = parser.parse_args()
    opt.mode = 'both'

    print(opt)

    if opt.task in ['val', 'test']:  # run normally
        test_merge(opt.data,
             opt.aug,
             opt.weights_helmet,
             opt.weights_alone,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.merge_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt,
             save_conf=opt.save_conf,
             agnostic=opt.agnostic,
             entity=opt.entity
             )

    elif opt.task == 'study':  # run over a range of settings and save/plot
        for weights in ['yolor-p6.pt', 'yolor-w6.pt', 'yolor-e6.pt', 'yolor-d6.pt']:
            f = 'study_%s_%s.txt' % (Path(opt.data).stem, Path(weights).stem)  # filename to save to
            x = list(range(320, 800, 64))  # x axis
            y = []  # y axis
            for i in x:  # img-size
                print('\nRunning %s point %s...' % (f, i))
                r, _, t = test(opt.data, weights, opt.batch_size, i,
                    opt.conf_thres, opt.iou_thres, opt.save_json, entity=opt.entity)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        # utils.general.plot_study_txt(f, x)  # plot
