import time
import cv2 as cv
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from pathlib import Path

# limit the number of cpus used by high performance libraries
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

import shutil

import sys
sys.path.insert(0, './object_detection')
from object_detection.detector import Detector

sys.path.insert(0, './object_tracking')
from object_tracking.tracker import Tracker

# sys.path.insert(0, './object_detection/yolov5')
from object_detection.yolov5.utils.datasets import LoadImages, LoadStreams
from object_detection.yolov5.utils.general import check_imshow
from object_detection.yolov5.utils.torch_utils import time_sync
from object_detection.yolov5.utils.plots import Annotator, colors 
from object_detection.yolov5.utils.general import LOGGER, check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh, increment_path

from configs.parser import get_config
PATH_CONFIGS = 'configs/manage.yaml'
cfg = get_config()
cfg.merge_from_file(PATH_CONFIGS)

def xywh_to_tlwh(bbox_xywh):
    if isinstance(bbox_xywh, np.ndarray):
        bbox_tlwh = bbox_xywh.copy()
    elif isinstance(bbox_xywh, torch.Tensor):
        bbox_tlwh = bbox_xywh.clone()
    bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
    bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
    return bbox_tlwh
    

def main():
    print('cuda is available ?', torch.cuda.is_available())
    ## TRACKING PARAMETER ##
    track_buffer = None
    
    # paremeter to run file main
    show_vid = cfg.MAIN.SHOW_VID
    webcam = cfg.MAIN.WEBCAM
    source = cfg.MAIN.PATH_SOURCE
    save_vid = cfg.MAIN.SAVE_VID
    visualize = cfg.MAIN.VISUALIZE
    out = cfg.MAIN.PATH_SAVE_FOLDER
    # evaluate = False
    
    # # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # # its own .txt file. Hence, in that case, the output folder is not restored
    # if not evaluate:
    #     if os.path.exists(out):
    #         pass
    #         shutil.rmtree(out)  # delete output folder
    #     os.makedirs(out)  # make new output folder
    
    # create Detector is Yolov5 classes=(2, 5, 6, 7, 8)
    detector = Detector(
        ckpt=cfg.MAIN.PATH_WEIGHT_YOLOV5, 
        conf_thres=cfg.MAIN.CONFIDENCE_THRESHOLD, 
        classes=cfg.MAIN.CLASSES
    )

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        source = '0'
        dataset = LoadStreams(source, img_size=detector.imgsz, stride=detector.stride, auto=detector.pt and not detector.jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=detector.imgsz, stride=detector.stride, auto=detector.pt and not detector.jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'
    
    dt, seen = [0.0, 0.0, 0.0], 0
    
    t0 = time.time()
    # Start Process
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        time_start = time.time()
        t1 = time_sync()
        img = torch.from_numpy(img).to(detector.device)
        img = img.half() if detector.half else img.float()
        img /= 255.0 # 0 - 255 to 0.0 - 1.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        t2 = time_sync()
        dt[0] += t2 - t1
        # Inference
        visualize = increment_path(save_path / Path(path).stem, mkdir=True) if visualize else False
        t3 = time_sync()
        dt[1] += t3 - t2
        
        pred, dt[1], dt[2], t3 = detector.detect(img=img, dt=dt, t2=t2, visualize=visualize)

        
        # all data in 1 frame
        for i, det in enumerate(pred):
            seen += 1
            
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            
            s += '%gx%g' % img.shape[2:] # ex. print string
            name_file = cfg.MAIN.NAME_SAVE_FILE
            save_path = str(Path(out) / name_file) 
            
            annotator = Annotator(im0, line_width=2, pil=not ascii)
            
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {detector.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                for *xyxy, conf, cls in reversed(det):
                    bbox = [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]
                    # print("BBOX : ", bbox)
    
                    
                    c = int(cls)  # integer class
                    # label = None if detector.hide_labels else (detector.names[c] if detector.hide_conf else f'{detector.names[c]} "sdsd" {conf.cpu(): .2f}')
                    if cfg.MAIN.SHOW_YOLO_RESULT:
                        label = f'{detector.names[c]} {conf: .2f}'
                        annotator.box_label(bbox, label, color=colors(c, True))
                        
                    if track_buffer is None:
                        track_buffer = [Tracker(bbox=bbox)]
                        # track_buffer.append(Tracker(bbox=bbox))
                        # tracker_buffer[0] = Tracker(xywhs, cfg.MAIN.MAX_AGE, cfg.MAIN.FRAME_BUFFER, cfg.MAIN.LIMIT_DISTANCE)
                    else:
                        for i, track_obj in enumerate(track_buffer):
                            center_current = track_obj.bboxes_to_center(bbox)
                            center_past = track_obj.center
                            distance_difference = track_obj.calculate_distance_p2p(center_current, center_past)
                            
                            # condition min distance to tracking
                            if distance_difference < cfg.MAIN.LIMIT_DISTANCE :
                                track_obj.update(bbox, cfg.MAIN.MAX_AGE, cfg.MAIN.FRAME_BUFFER, cfg.MAIN.MIN_DISTANCE_UPDATE)
                            else:
                                flag_new_object = True
                        if flag_new_object is True:
                            track_buffer.append(Tracker(bbox=bbox))
                            flag_new_object = False
                                    
            else:
                pass
            
            if(len(track_buffer) > 1):
                for i, track_obj in enumerate(track_buffer):
                    if()
            # If want draw outer process but is unnecessary loop because can draw in process / use for develop debug
            
        # for i, track_obj in enumerate(track_buffer):
        #     b1 = track_obj.path_bbox[0]
        #     b2 = track_obj.path_bbox[track_obj.frame_buffer]
        #     p1 = track_obj.bboxes_to_center(b1)
        #     p2 = track_obj.bboxes_to_center(b2)
        #     print("p1 : ", p1)
        #     print("p2 : ", p2)
        #     cv.line(im0, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 5)
                    
                    
            # for i in analyzer.buffer_list:
            #     buffer = analyzer.buffer_list[i]
            #     bbox = buffer.bbox
            #     c = int(buffer.class_id)  # integer class
            #     label = f'{buffer.track_id} {detector.names[c]} {buffer.conf:.2f} id : {c}'
            #     if analyzer.show_yolo_result is False :
            #         annotator.box_label(bbox, label, color=colors(c, True))
                    
            #     if(buffer is not None and buffer.angle is not None):
            #         cv.putText(im0, "angle : {:.2f}".format(buffer.angle), (bbox[0],bbox[1] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
                
            #     # draw line path of objective movement
            #     if((buffer.frame.get(1) is not None ) and buffer.frame.get(buffer.max_buffer) is not None):
            #         x1, y1 = buffer.frame[1]
            #         x2, y2 = buffer.frame[buffer.max_buffer]
            #         # color = (0, 255, 0)
            #         center_x, center_y = buffer.center
            #         label = f'{buffer.track_id} '
            #         box_draw = [
            #             center_x - analyzer.length_check_polygon, 
            #             center_y - analyzer.length_check_polygon, 
            #             center_x + analyzer.length_check_polygon, 
            #             center_y + analyzer.length_check_polygon
            #         ]
            #         annotator.box_label(box_draw, label, color=(100,255,0))
            #         cv.line(im0, (x1,y1), (x2, y2), (0, 255, 0), 5)
                

            
            # Print time (inference-only)
            # LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')
            
            # Stream results
            im0 = annotator.result()
            
            
            # show data in image
            fps = 1./(time.time()-time_start)
            cv.putText(im0, "FPS: {:.2f}".format(fps), (5,30), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
            
                    
            
        if show_vid:
            cv.imshow(p, im0)
            if cv.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

        # Save results (image with detections)
        if save_vid:
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv.VideoWriter):
                    vid_writer.release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path += '.mp4'

                vid_writer = cv.VideoWriter(save_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer.write(im0)
    
    print('Done. (%.3fs)' % (time.time() - t0))

                
    
    
if __name__ == "__main__":
    main()