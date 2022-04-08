import sys
sys.path.insert(0, './object_detection/yolov5')
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.general import check_img_size, non_max_suppression


import time
# import cv2 as cv
import torch
# import torch.backends.cudnn as cudnn
import math
import numpy as np


class Detector():
    def __init__(
        self, 
        ckpt='yolov5/weights/yolov5s.pt', 
        imgsz=640, device='0', 
        max_det=1000,
        augment=False, 
        conf_thres=0.4, 
        iou_thres=0.5, 
        classes=None, 
        agnostic_nms=False, 
        half=False, 
        dnn=False, 
        hide_labels=False,
        hide_conf=False,
    ):
        # set size images
        imgsz = [imgsz, imgsz]
        # Initialize
        self.device = select_device(device)
        self.half = half
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA
        
        
        # print("IMGSZ: ", imgsz)
        # Load model
        self.model = DetectMultiBackend(weights=ckpt, device=self.device, dnn=dnn)
        self.stride, self.names, self.pt, self.jit, self.onnx = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx
        self.imgsz = check_img_size(imgsz=imgsz, s=self.stride)  # check img_size
        # print("self.IMGSZ: ", self.imgsz)
        # Half
        self.half &= self.pt and self.device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if self.pt:
            self.model.model.half() if half else self.model.model.float()
            
        #  Parameter Detection
        self.augment = augment
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        
        # Paremeter accessory
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf,        
        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        # Run inference
        if self.pt and self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.imgsz).to(self.device).type_as(next(self.model.model.parameters())))  # warmup
            # self.model.warmup(imgsz=(1, 3, *self.imgsz), half=self.half)  # warmup
        
    def detect(self, img, dt, t2, visualize=False):
        pred = self.model(img, augment=self.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2
        # Apply NMS
        pred = non_max_suppression(
            # pred, conf_thres, iou_thres, agnostic=agnostic_nms)
            pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms, max_det=self.max_det)
        dt[2] += time_sync() - t3
        return pred, dt[1], dt[2], t3