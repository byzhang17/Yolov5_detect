# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import platform
import sys
from pathlib import Path
import math
import torch
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

from flask import Flask, jsonify, request

app = Flask(__name__)

path_light = sys.argv[1]  # 灯光的wight路径
path_switch = sys.argv[2]  # 开关wight存放路径
path_number = sys.argv[3]  # 指针wight存放路径
path_pointer = sys.argv[4]  # 数字wight存放路径


# path_result = sys.argv[5]  # 结果存放路径


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def xywh2xyxy(x):
    y = x.copy()
    y[0] = x[0] - x[2] / 2
    y[1] = x[1] - x[3] / 2
    y[2] = x[0] + x[2] / 2
    y[3] = x[1] + x[3] / 2
    return y


def line_k(x1, y1, x2, y2):
    if x1 == x2:
        k = 10000
    else:
        k = (y2 - y1) / (x2 - x1)
    return k


def get_cobb(img):
    W = img.shape[1]
    H = img.shape[0]
    k1 = 0
    k2 = 0
    Cobb = 0
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray_img, 80, 255, cv2.THRESH_BINARY)
    tm = thresh1.copy()
    test_main = tm[int(W / 25):int(W * 23 / 25), int(H / 25):int(H * 23 / 25)]
    edges = cv2.Canny(test_main, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 40)
    if lines is not None:
        lines = lines[:, 0, :]
        result = edges.copy()
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + result.shape[1] * (-b))
            y1 = int(y0 + result.shape[1] * a)
            x2 = int(x0 - result.shape[0] * (-b))
            y2 = int(y0 - result.shape[0] * a)
            # 指针
            if y2 >= H / 2 and y2 <= H * 4 / 5 and x2 >= H / 8:
                k2 = line_k(x1, y1, x2, y2)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        Cobb = int(math.fabs(np.arctan((k1 - k2) / (float(1 + k1 * k2))) * 180 / np.pi) + 0.5)
    return Cobb


class DetectAPI:
    def __init__(self, weights='weights/yolov5s.pt', source='data/images', data='data/coco128.yaml', imgsz=None,
                 conf_thres=0.55,
                 iou_thres=0.45, max_det=1000, device='cpu', view_img=False, save_txt=False,
                 save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False,
                 visualize=False, update=False, project='D:/code/yolov5-v8.0/runs/detect', name='myexp', exist_ok=False,
                 line_thickness=1,
                 hide_labels=False, hide_conf=False, half=False, dnn=False, vidstride=1):

        if imgsz is None:
            self.imgsz = [640, 640]
        self.weights = weights
        self.data = data
        self.source = source
        self.imgsz = [640, 640]
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.view_img = view_img
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.save_crop = save_crop
        self.nosave = nosave
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.visualize = visualize
        self.update = update
        # self.project = project
        self.name = name
        self.exist_ok = exist_ok
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.half = half
        self.dnn = dnn
        self.vidstride = vidstride

        device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data=self.data, fp16=self.half)

    def run(self, source, project):
        save_dir1 = increment_path(Path(project) / self.name, exist_ok=self.exist_ok)  # increment run
        (save_dir1 / 'labels' if self.save_txt else save_dir1).mkdir(parents=True, exist_ok=True)  # make dir
        save_dir = save_dir1
        source = str(source)
        save_img = not self.nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download
        txts = []
        crops = []
        save_path = ''
        txt_path = ""
        mylabel = []
        # Directories
        # save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
        # (save_dir / 'labels' if self.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        # device = select_device(self.device)
        # model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data=self.data, fp16=self.half)
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        if webcam:
            self.view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=self.vidstride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=self.vidstride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        self.model.warmup(imgsz=(1 if pt or self.model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if self.visualize else False
                pred = self.model(im, augment=self.augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                           max_det=self.max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results

                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                        txts.append((int(cls), *xywh))
                        if self.save_txt:  # Write to file
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or self.save_crop or self.view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if self.hide_labels else (
                                names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                            mylabel.append(str(label))
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if self.save_crop:
                            crop, f = save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                                                   BGR=True)
                            crops.append((crop, f))

                # Stream results
                im0 = annotator.result()
                if self.view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if self.save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if self.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if self.update:
            strip_optimizer(self.weights[0])  # update model (to fix SourceChangeWarning)

        return txts, crops, save_path, txt_path, save_dir1


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp21/weights/best.pt',
                        help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'datasets/pointer/images/val',
                        help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/pointer.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_false', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_false', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


detect_light = DetectAPI(weights=path_light, save_txt=True)
detect_switch = DetectAPI(weights=path_switch, save_txt=True)
detect_number = DetectAPI(weights=path_number, save_txt=True, save_crop=True)
detect_pointer = DetectAPI(weights=path_pointer, save_txt=True)


@app.route('/predictlight', methods=['POST'])
def predict_light():
    if request.method == 'POST':
        image_path = request.json['path']  # 图片路径
        result_path = request.json['result']

        # 调用检测函数
        # 1.灯光检测
        txt, crops, save_path, txt_path, save_dir = detect_light.run(source=image_path, project=result_path)
        # json_res = image_to_base64("./serve_res.png")
        # 2.开关检测
        txt, crops, save_path, txt_path, save_dir = detect_switch.run(source=save_path, project=result_path)

        deal_path = txt_path + '.txt'

        new_txt = sorted(txt, key=lambda d: d[2])

        with open(deal_path, "a") as f:
            f.truncate(0)
        while new_txt:
            txt1 = []
            for i in range(0, len(new_txt)):
                if i == len(new_txt) - 1:
                    txt1 = new_txt
                    new_txt = []
                    break
                else:
                    if new_txt[i + 1][2] - new_txt[i][2] > 0.05:
                        txt1 = new_txt[0:i + 1]
                        new_txt = new_txt[i + 1:]
                        break
            txt1 = sorted(txt1, key=lambda d: d[1])
            with open(deal_path, "a") as f:
                for text in txt1:
                    f.write(('%g ' * len(text)).rstrip() % text + '\n')
        # 3.指针和数字检测
        txts, crops, save_path3, txt_path3, save_dir_number = detect_number.run(source=save_path, project=result_path)

        print(save_path)
        dict = {}
        for i in range(0, len(txts)):
            dict[txts[i]] = crops[i]
        path1 = str(save_dir_number) + '/num_labels/'
        path2 = str(save_dir_number) + '/pointer_labels/'
        os.mkdir(path1)
        os.mkdir(path2)
        path3 = path1 + str(os.path.basename(save_path)).split('.')[0] + '.txt'
        path4 = path2 + str(os.path.basename(save_path)).split('.')[0] + '.txt'
        open(path3, 'a')
        open(path4, 'a')
        txt1 = []
        txt2 = []
        for txt in txts:
            if txt[0] == 0:
                txt1.append(txt)
            else:
                txt2.append(txt)
        new_txt1 = []
        new_txt2 = []
        new_txt1 = sorted(txt1, key=lambda d: d[2])
        new_txt2 = sorted(txt2, key=lambda d: d[2])

        if len(new_txt1):
            while new_txt1:
                texts1 = []
                for i in range(0, len(new_txt1)):
                    if i == len(new_txt1) - 1:
                        texts1 = new_txt1
                        new_txt1 = []
                        break
                    else:
                        if new_txt1[i + 1][2] - new_txt1[i][2] > 0.05:
                            texts1 = new_txt1[0:i + 1]
                            new_txt1 = new_txt1[i + 1:]
                            break
                texts1 = sorted(texts1, key=lambda d: d[1])
                for text1 in texts1:
                    txt3, crop3, save_path3_n, txt_path3_n, save_dir_number_n = detect_pointer.run(
                        source=dict[text1][1], project=result_path)
                    new_txt3 = []
                    new_txt3 = sorted(txt3, key=lambda d: d[2])
                    while new_txt3:
                        texts2 = []
                        for i in range(0, len(new_txt3)):
                            if i == len(new_txt3) - 1:
                                texts2 = new_txt3
                                new_txt3 = []
                                break
                            else:
                                if new_txt3[i + 1][2] - new_txt3[i][2] > 0.05:
                                    texts2 = new_txt3[0:i + 1]
                                    new_txt3 = new_txt3[i + 1:]
                                    break
                        texts2 = sorted(texts2, key=lambda d: d[1])
                        with open(path3, "a") as f:
                            for text2 in texts2:
                                f.write(str(text2[0]))
                            f.write(' ' + ('%g ' * len(text1)).rstrip() % text1 + '\n')

        if len(new_txt2):
            while new_txt2:
                texts3 = []
                for i in range(0, len(new_txt2)):
                    if i == len(new_txt2) - 1:
                        texts3 = new_txt2
                        new_txt2 = []
                        break
                    else:
                        if new_txt2[i + 1][2] - new_txt2[i][2] > 0.1:
                            texts3 = new_txt2[0:i + 1]
                            new_txt2 = new_txt2[i + 1:]
                            break
                texts3 = sorted(texts3, key=lambda d: d[1])
                with open(path4, "a") as f:
                    for text in texts3:
                        cobb = get_cobb(dict[text][0])
                        f.write(str(cobb) + ' ' + ('%g ' * len(text)).rstrip() % text + '\n')

        f1 = open(path3, 'r')
        f2 = open(path4, 'r')

        datas1 = f1.readlines()
        l1 = []
        flag = [0, 0, 0, 0]
        string = ''
        for data in datas1:
            a = data.strip('\n').split(' ')
            xywh = [float(a[2]), float(a[3]), float(a[4]), float(a[5])]
            if xywh == flag:
                string = string + a[0] + ' '
            else:
                l1.append((string, flag))
                flag = xywh
                string = a[0] + ' '
        l1.append((string, flag))

        datas2 = f2.readlines()
        l2 = []
        flag = [0, 0, 0, 0]
        string = ''
        for data in datas2:
            a = data.strip('\n').split(' ')
            xywh = [float(a[2]), float(a[3]), float(a[4]), float(a[5])]
            if xywh == flag:
                string = string + a[0] + ' '
            else:
                l2.append((string, flag))
                flag = xywh
                string = a[0] + ' '
        l2.append((string, flag))

        img = cv2.imread(save_path)
        size = img.shape
        w = size[1]
        h = size[0]
        if len(l1) > 1:
            for i in range(1, len(l1)):
                string = l1[i][0]
                xywh = l1[i][1]
                xyxy = xywh2xyxy(xywh)
                x1 = int(xyxy[0] * w)
                y1 = int(xyxy[1] * h)
                x2 = int(xyxy[2] * w)
                y2 = int(xyxy[3] * h)
                p1 = (x1, y1)
                p2 = (x2, y2)
                cv2.rectangle(img, p1, p2, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                tf = max(2 - 1, 1)  # font thickness
                w1, h1 = cv2.getTextSize(string, 0, fontScale=2 / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h1 >= 3
                p2 = p1[0] + w1, p1[1] - h1 - 3 if outside else p1[1] + h1 + 3
                cv2.putText(img,
                            string, (p1[0], p1[1] - 2 if outside else p1[1] + h1 + 2),
                            0,
                            2 / 3,
                            (0, 0, 255),
                            thickness=tf,
                            lineType=cv2.LINE_AA)

        if len(l2) > 1:
            for i in range(1, len(l2)):
                string = l2[i][0]
                xywh = l2[i][1]
                xyxy = xywh2xyxy(xywh)
                x1 = int(xyxy[0] * w)
                y1 = int(xyxy[1] * h)
                x2 = int(xyxy[2] * w)
                y2 = int(xyxy[3] * h)
                p1 = (x1, y1)
                p2 = (x2, y2)
                cv2.rectangle(img, p1, p2, (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
                tf = max(3 - 1, 1)  # font thickness
                w1, h1 = cv2.getTextSize(string, 0, fontScale=3 / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h1 >= 3
                p2 = p1[0] + w1, p1[1] - h1 - 3 if outside else p1[1] + h1 + 3
                cv2.putText(img,
                            string, (p1[0], p1[1] - 2 if outside else p1[1] + h1 + 2),
                            0,
                            3 / 3,
                            (0, 255, 0),
                            thickness=tf,
                            lineType=cv2.LINE_AA)

        cv2.imwrite(str(save_dir_number) + '/' + str(os.path.basename(save_path)), img)

        json1 = jsonify({'txt': txt, 'save_path': save_path3, 'txt_path': txt_path})
    # jsonify中保存着结果图片的base编码，拿下来客户端解码即可得到结果图片
    return json1


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    # run(**vars(opt))


if __name__ == "__main__":
    # opt = parse_opt()
    # main(opt)
    app.run(debug=True, host='127.0.0.1', port=3333)
