from os import name
import os
import torch
import cv2 as cv
import time
import numpy as np
import math
from tools.point import Point, doIntersect
from shapely.geometry import Polygon
from datetime import datetime
import csv
import argparse
import os
import sys
from pathlib import Path

from influxdb import InfluxDBClient


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import socketio 
import base64

sio = socketio.Client()
# sio.connect('http://localhost:45678')
sio.connect("http://localhost:4333") 

from configs.parser import get_config
# PATH_CONFIGS = 'configs/manage1.yaml'
# cfg = get_config()
# cfg.merge_from_file(PATH_CONFIGS)





############ FUNCTIONAL ############################################
# LIST TYPE OBJECT DETECTION
def list_obj_detection(name):
    if(name == 'car' or name == 'bus' or name == 'motorcycle' or name == 'train' or name == 'truck'):
        return True
    else:
        return False

# SELECT COLOR FOR OBJECTIVE
def select_color_object(name):
    # COLOR OBJECTIVES
    color_car_detection = (0, 255, 0)
    color_car_tracking = (255, 255, 0)

    color_truck_detection = (0, 0, 255)
    color_truck_tracking = (127, 127, 127)

    color_bus_detection = (255, 0, 0)
    color_bus_tracking = (255, 127, 127)

    color_motorcycle_detection = (255, 0, 0)
    color_motorcycle_tracking = (255, 127, 127)
    
    if(name == 'car'):
        return color_car_detection, color_car_tracking
    if(name == 'bus'):
        return color_bus_detection, color_bus_tracking
    # if(name == 'truck'):
    #     return color_truck_detection, color_truck_tracking
    if(name == 'motorcycle'):
        return color_motorcycle_detection, color_motorcycle_tracking
    else:
        return (0, 255, 0), (0, 255, 255)

def bboxes_to_center(box):
    center_x = box[0] + (abs(box[2] - box[0])//2)
    center_y = box[1] + (abs(box[3] - box[1])//2)
    return [center_x, center_y]

def calculate_distance_p2p(p1, p2):
    a = np.array((p1))
    b = np.array((p2))
    return np.linalg.norm(a-b)

# DRAWING BOUNDING BOXES
def draw_box_obj(color, center_x, center_y, d_xmin, d_xmax, d_ymin, d_ymax, text, frame):
    cv.circle(frame,(center_x, center_y), 1, (0,0,255), 2)
    cv.rectangle(frame, (d_xmin, d_ymin), (d_xmax, d_ymax), color, 2)
    cv.putText(frame, text, (d_xmin, d_ymin-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (color), 2)

# UPDATE POSITION LIFE POINT OF COUNTING
def update_tracking_obj(index, center_x, center_y, distance, tracking_list_obj, life, max_frame_buffer, min_distance_update, frame, bbox):
        tracking_list_obj[index][0][0] = center_x
        tracking_list_obj[index][0][1] = center_y
        tracking_list_obj[index][1] = life
        tracking_list_obj[index][2] = True
        tracking_list_obj[index][3] += 1
        tracking_list_obj[index][4][0] += 1
        tracking_list_obj[index][4][1] += distance
        tracking_list_obj[index][8] = bbox
        
        # save_path to make sure moving and save path like window
        id_frame = tracking_list_obj[index][6]

        # print("id : ", id_frame)
        p1 = tracking_list_obj[index][5][id_frame]
        p2 = [center_x, center_y]
        distance = calculate_distance_p2p(p1, p2)
        if(id_frame < max_frame_buffer):
            if (distance > min_distance_update):
                tracking_list_obj[index][5].append([center_x, center_y])
                tracking_list_obj[index][6] += 1         
        else:
            # print("jello")
            if (distance > min_distance_update):
                # shift frame like window
                r = max_frame_buffer
                for i in range(0, r):
                    # print("i : ", i)
                    id_update = i+1
                    id_dest = i+2
                    if(id_update == max_frame_buffer):
                        # print("geeldddddddddddddddddddd")       
                        tracking_list_obj[index][5][id_update] = [center_x, center_y]
                    else:
                        # print("geel")
                        center_next = tracking_list_obj[index][5][id_dest]
                        tracking_list_obj[index][5][id_update] = center_next
            



# UPDATE SPEED OBJECTIVE AND RESET DISTANCE STACK
def update_speed_object(index, speed, tracking_list_obj):
    tracking_list_obj[index][4][1] = 0
    tracking_list_obj[index][4][2] = speed


# CALCULATE DISTANCE OBJECTIVE
def calculate_distance(x1, y1, x2, y2):
    # return math.sqrt(((abs(x2 - x1))**2) + ((abs(y2 - y1))**2))
    distance = math.sqrt(((abs(x2 - x1))**2) + ((abs(y2 - y1))**2))
    return distance
    

# CALCULATE SPEED IN PIXEL / MINUS UNIT OBJECTIVE
def calculate_speed(d_pixel, fps_sp, meters_per_pixels):

    # pixels per seconds
    # speed_per_pixel_second  = d_pixel * fps_sp
    # return speed_per_pixel

    # pixels per hours
    # speed_per_pixel_hour  = d_pixel * fps_sp * 3.6
    # return speed_per_pixel_hour

    # meters per seconds
    # speed_per_meters_second  = d_pixel * METERS_PER_PIXELS * fps_sp
    # return speed_per_meters_second
    
    # kilometer per hour 
    speed_per_meters_km_hr  = d_pixel * meters_per_pixels * fps_sp * 3.6
    return speed_per_meters_km_hr


# CALCULATE AVERAGE SPEED OBJECTIVE
def estimate_speed(spd, frame_to_estimate):
    speed = spd / frame_to_estimate
    return speed

def estimate_speed2(index, d_pixel, fps_sp, frame_to_cal, tracking_list_obj, meters_per_pixels, speed_out_of_sight_car_stop):

    # pixels per seconds
    # speed_per_pixel_second  = (d_pixel/frame_to_cal) * fps_sp
    # return speed_per_pixel

    # kilopixels per hours
    # speed_per_pixels_per_hr  = (d_pixel/(frame_to_cal)) * fps_sp * 3600 * (1/1000)
    # if (tracking_list_obj[index][4][2] > 0 ): 
    #     if(speed_per_pixels_per_hr <= speed_out_of_sight_car_stop):
    #         speed_per_pixels_per_hr = 0

    # meters per seconds
    # speed_per_meters_second  = (d_pixel/frame_to_cal) * METERS_PER_PIXELS * fps_sp
    # return speed_per_meters_second
    
    # kilometer per hour 
    speed_per_meters_km_hr  = (d_pixel/(frame_to_cal)) * meters_per_pixels * fps_sp * 3600 * (1/1000)
    if (tracking_list_obj[index][4][2] > 0 ): 
        if(speed_per_meters_km_hr <= speed_out_of_sight_car_stop):
            speed_per_meters_km_hr = 0
        # else:
        #     estimate_cal = abs(speed_estimate_obj - tracking_list_obj[index][4][2])
        #     if(estimate_cal > SPEED_OUT_OF_SIGHT ):
        #         speed_per_meters_km_hr = tracking_list_obj[index][4][2]


    return speed_per_meters_km_hr
    

    

###################################################################

def main(opt):
    cfg = get_config()
    print('config path : ', opt.config)
    cfg.merge_from_file(opt.config)
    
    
    torch.cuda.is_available()

    ############ CONFIGURE PARAMETERS #################################
    # FLAGE EVENT
    flag_config_ROI = False


    ## TRACKING PARAMETERS
    # list parameter definition
    # tracking_list_obj = [
    #     (obj_center_x, obj_center_y), 
    #     life for delete tracking, 
    #     flag for decrease life, 
    #     number for count car,
    #     (number for start calculate speed, distance stack, speed is pixels(or kilimeters) per hours)
    #     [[center_buffer]],
    #     limit_number,
    #     flag_count,
    #     count_check_accident
    #     flag_accident
    # ] 
    
    tracking_list_obj = None  
    tracking_list_delete_obj = []

    # configure tracking
    LIFE = cfg.MAIN.MAX_AGE
    # tracking_x_distance = 30
    # tracking_y_distance = 30

    # configure for count car if car tracked x round
    # FRAME_TO_COUNT = 8

    # configure for calculate speed car
    FRAME_TO_CALCULATE_SPEED = cfg.MAIN.FRAME_TO_CALCULATE_SPEED
    SPEED_OUT_OF_SIGHT  = cfg.MAIN.SPEED_OUT_OF_SIGHT # Km/hr
    SPEED_OUT_OF_SIGHT_CAR_STOP = cfg.MAIN.SPEED_OUT_OF_SIGHT_CAR_STOP # Km/hr

    # meter per pixels need to calculate from standart road with camera  meter/pixels
    # METERS_PER_PIXELS  = 0.04136914426
    METERS_PER_PIXELS  = cfg.MAIN.METERS_PER_PIXELS
    # METERS_PER_PIXELS  = 1/8.8
    # METERS_PER_PIXELS = 3/35

    # METERS_PER_PIXELS = 0.25
    FRAME_RATE_SOURCE = cfg.MAIN.FRAME_RATE_SOURCE


    # parameter for get time
    count_time_frame = 0

    # OUTPUTS
    count_car = 0
    count_car_2 = 0
    count_accident = 0
    count_vehicle_all = 0
    
    speed_average_flow = 0 # km/hr
    
    check_header = 0
    fps_avg = 0
    process_time_minute = 0
    process_time_hours = 0
    process_time_day = 1
    process_time_string = 'day1-00.00'
    
    header = cfg.MAIN.HEADER_CSV
    


    ###################################################################
    # RUN MODEL 
    model = torch.hub.load('C:/project/traffic-analysis-main/object_detection/yolov5', 'custom', path=cfg.MAIN.PATH_WEIGHT_YOLOV5, source='local') 
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
    # 

    cap = cv.VideoCapture(cfg.MAIN.PATH_SOURCE)
    
    ret, image_origin = cap.read()
    if(image_origin is not None):
        image = cv.resize(image_origin, (960, 540))
    else:
        print('image is None pls check your datasource ')
        os._exit(0) 
          

    if cfg.MAIN.SAVE_VID:
        # SAVE VIDEO CONFIG
        PATH_SAVE = cfg.MAIN.PATH_SAVE
        PATH_SAVE_IMAGE_TO_WEB = 'D:/Project-web/t1/src/static/cctv01.jpg'
        # (vdo_width) = (int(cap.get(3)))
        # (vdo_height) = (int(cap.get(4)))
        vdo_width = 960
        vdo_height = 540
        video = cv.VideoWriter(PATH_SAVE, cv.VideoWriter_fourcc(*'mp4v'), 30, (vdo_width, vdo_height))
        # video2 = cv.VideoWriter(PATH_SAVE, cv.VideoWriter_fourcc(*'mp4v'), 30, (vdo_width, vdo_height))

    # select ROI 
    if flag_config_ROI is True:
        roi_xmin, roi_ymin, x, y = cv.selectROI("AREADETECTION", image, False)
        roi_xmax = roi_xmin + x
        roi_ymax = roi_ymin + y
        print("ROI (xmin, ymin, xmax, ymax)")
        print(roi_xmin, roi_ymin, roi_xmax, roi_ymax)
        cv.destroyAllWindows()
    else:
        area = cfg.MAIN.AREA_DETECTION
        # roi_xmin = 131
        # roi_ymin = 152
        # roi_xmax = 789
        # roi_ymax = 530
        
        roi_xmin, roi_ymin, roi_xmax, roi_ymax = area[0], area[1], area[2], area[3]

        # roi_xmin = 275
        # roi_ymin = 66
        # roi_xmax = 886
        # roi_ymax = 504

    # frame_length = 0

    # Start Process
    while True:
        ret, frame_origin = cap.read()
        if frame_origin is None:
            print('Complete')
            break
        frame = cv.resize(frame_origin, (960, 540))
        full_frame = frame.copy()

        time_start = time.time()
        if flag_config_ROI is True:
            frame = full_frame[roi_ymin: roi_ymax, roi_xmin: roi_xmax]
            cv.rectangle(full_frame, (roi_xmin,roi_ymin), (roi_xmax, roi_ymax), (0, 255, 255), 2)
            cv.putText(full_frame, "AREA DETECTION", (roi_xmin, roi_ymin-5) , cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        else:
            frame = full_frame[roi_ymin: roi_ymax, roi_xmin: roi_xmax]
            cv.rectangle(full_frame, (roi_xmin,roi_ymin), (roi_xmax, roi_ymax), (0, 255, 255), 2)
            cv.putText(full_frame, "AREA DETECTION", (roi_xmin, roi_ymin-5) , cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        
        results = model(frame, size=640)  # includes NMS
        
        # draw line for count
        # arr_count_line = cfg.MAIN.COUNT_LINE
        # for obj in arr_count_line:
        #     p1, p2 = obj[0], obj[1]
        #     cv.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (200, 0, 0), 5)
            
        
        arr_count_area = cfg.MAIN.COUNT_AREA
        for area in arr_count_area:
            # Define ps = [(400,500),(600,200), (800, 200),(750,500),(600,580)]
            # print("POLY", ps)
            arr_ps = np.array(area)
            cv.polylines(frame, [arr_ps], True, (255, 0, 0), thickness=2)
            cv.putText(frame, "AREA COUNTING ", (area[0][0],area[0][1] + 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            # cv.rectangle(frame, (area[0], area[1]), (area[2], area[3]), (255, 0, 0), 2)
            
        arr_accident_area = cfg.MAIN.AREA_ACCIDENT_CHECK
        for area in arr_accident_area:
            arr_ps = np.array(area)
            cv.polylines(frame, [arr_ps], True, (0, 0, 255), thickness=2)
            cv.putText(frame, "AREA ACCEDENT DETECT ", (area[0][0],area[0][1] + 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        
        # Results
        # results.print()
        # print(results.pandas().xyxy[0])  # img1 predictions (pandas)

        for idx, obj in results.pandas().xyxy[0].iterrows():

            if(obj['confidence'] > cfg.MAIN.CONFIDENCE_THRESHOLD and list_obj_detection(obj['name']) ):
                # define parameters
                (xmin, ymin) = (int(obj['xmin']), int(obj['ymin']))
                (xmax, ymax) = (int(obj['xmax']), int(obj['ymax']))
                (conf, name) = (obj['confidence'], obj['name'])
                (x_length) = (abs(xmax - xmin))
                (y_length) = (abs(ymax - ymin))
                # print(xmin, ymin, xmax, ymax)
                # print(x_length, y_length)
                (obj_center_x, obj_center_y) = (int(xmin + (x_length//2)), int(ymin + (y_length//2)))
                color_detection, color_tracking = select_color_object(name)
                # print("CDETECTION", color_detection, "CTRACKING", color_tracking)

                if tracking_list_obj is None:
                    
                    # get first new obj to tracking
                    tracking_list_obj = []
                    tracking_list_obj.append([
                        [obj_center_x, obj_center_y], 
                        LIFE, 
                        True, 
                        0, 
                        [0, 0, 0],
                        [[obj_center_x, obj_center_y]], 
                        0, 
                        False, 
                        [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], 
                        0,
                        False
                    ])

                    # draw bounding new box obj
                    text_new_object = "New " + name + ' : ' + "{:.2f}".format(conf) + ' %'
                    draw_box_obj(color_detection, obj_center_x, obj_center_y, xmin, xmax, ymin, ymax, text_new_object, full_frame)
                else:
                    # check id old object ?
                    for i, track_obj in enumerate(tracking_list_obj):
                        speed_obj = 0
                        distance_obj = 0
                        # print("track objective", track_obj)
                        
                        # count to check accident check from speed < 1 km/hr
                        for area in arr_accident_area:
                            polygon_accident = Polygon(area)
                            polygon_obj = Polygon(tracking_list_obj[i][8])
                            intersection_accident = polygon_accident.intersects(polygon_obj)
                            if intersection_accident:
                                if(tracking_list_obj[i][4][2] < cfg.MAIN.MIN_SPEED_TO_CHECK_COUNT_ACCIDENT):
                                    tracking_list_obj[i][9] += 1
                                else:
                                    tracking_list_obj[i][9] = 0
                                    tracking_list_obj[i][10] = False
                                    
                            # get average speed flow
                            if(tracking_list_obj[i][4][2] > cfg.MAIN.MIN_SPEED_TO_CHECK_COUNT_ACCIDENT):
                                if(speed_average_flow == 0):
                                    speed_average_flow = tracking_list_obj[i][4][2]
                                else:
                                    speed_average_flow = (speed_average_flow + tracking_list_obj[i][4][2])/2
                                    
                        
                        # check accident
                        accident_frame_check = tracking_list_obj[i][9]
                        check_accident_frame = cfg.MAIN.LIMIT_TIME * cfg.MAIN.FRAME_RATE_SOURCE * 60
                        if(accident_frame_check > check_accident_frame and tracking_list_obj[i][10] is False):
                            count_accident += 1
                            tracking_list_obj[i][10] = True

                        # calculate distance per frame
                        distance_obj = calculate_distance_p2p((obj_center_x, obj_center_y), (track_obj[0][0], track_obj[0][1]))
                        # print("DISTANCE BEETWEEN 2 POINT: ", distance_obj)

                        # if(abs(obj_center_x - track_obj[0][0]) < tracking_x_distance and abs(obj_center_y - track_obj[0][1]) < tracking_y_distance):
                        if(distance_obj <= cfg.MAIN.LIMIT_DISTANCE):

                            # calculate speed in one frame
                            # speed_obj = calculate_speed(distance_obj, FRAME_RATE_SOURCE)

                            # update position life countCondition and distance objective tracked
                            update_tracking_obj(i, obj_center_x, obj_center_y, distance_obj, tracking_list_obj, LIFE, cfg.MAIN.FRAME_BUFFER, cfg.MAIN.MIN_DISTANCE_UPDATE, frame, [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
                            
                            # draw path and check to count
                            idx_frame_buffer = tracking_list_obj[i][6]
                            if(idx_frame_buffer > 1):
                                p1 = tracking_list_obj[i][5][1]
                                p2 = tracking_list_obj[i][5][idx_frame_buffer]
                                cv.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 5)
                                
                                if (tracking_list_obj[i][3] > cfg.MAIN.FRAME_TO_COUNT):  
                                    for area in arr_count_area:
                                        poly1 = Polygon(area)
                                        poly2 = Polygon(tracking_list_obj[i][8])
                                        intersection = poly1.intersects(poly2)
                                        if(intersection):
                                            cv.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 5)
                                            if(tracking_list_obj[i][7] is False):
                                                count_car += 1
                                                count_vehicle_all += 1
                                                tracking_list_obj[i][7] = True
                                    
                                # for obj in arr_count_line:
                                    
                                    
                                #     print(p1.intersects(p2))
                                #     pp1 = Point(obj[0][0], obj[0][1])
                                #     qq1 = Point(obj[1][0], obj[1][1])
                                #     pp2 = Point(tracking_list_obj[i][5][1][0], tracking_list_obj[i][5][1][1])
                                #     qq2 = Point(tracking_list_obj[i][5][idx_frame_buffer][0], tracking_list_obj[i][5][idx_frame_buffer][1])
                                    
                                #     intersection = doIntersect(pp1, qq1, pp2, qq2)
                                    
                                #     if(intersection):
                                #         cv.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 5)
                                #         if(tracking_list_obj[i][7] is False):
                                #             count_car += 1
                                #             tracking_list_obj[i][7] = True
                                        
                                            
                                    
                            
                            # condition to counting object
                            if (track_obj[3] == cfg.MAIN.FRAME_TO_COUNT):
                                count_car_2 += 1
                            
                            # condition to estimate speed object
                            if((track_obj[4][0] % FRAME_TO_CALCULATE_SPEED) == 0 ):
                                speed_estimate_obj = estimate_speed2(i, track_obj[4][1], FRAME_RATE_SOURCE, FRAME_TO_CALCULATE_SPEED, tracking_list_obj, METERS_PER_PIXELS, SPEED_OUT_OF_SIGHT_CAR_STOP)
                                update_speed_object(i, speed_estimate_obj, tracking_list_obj)

                            # draw bounding box for tracking object
                            if(tracking_list_obj[i][10] is True):
                                text_accident = name + ' : ' + "Accident"
                                draw_box_obj((0, 0, 255), obj_center_x, obj_center_y, xmin, xmax, ymin, ymax, text_accident, frame=frame)
                            else:
                                text_tracking_object = "Tracking " + name + ' : ' + "{:.2f}".format(conf) + ' %' + " {:.2f}".format(track_obj[4][2])
                                draw_box_obj(color_tracking, obj_center_x, obj_center_y, xmin, xmax, ymin, ymax, text_tracking_object, frame=frame)
                            # show accident
                            
                            flag_new_object = False
                            break
                        else:
                            flag_new_object = True
                            tracking_list_obj[i][2] = False
                    
                    if flag_new_object is True:
                        # new objective so adding list tracking
                        tracking_list_obj.append([
                            [obj_center_x, obj_center_y], 
                            LIFE, 
                            True, 
                            0, 
                            [0, 0, 0],
                            [[obj_center_x, obj_center_y]], 
                            0, 
                            False, 
                            [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], 
                            0,
                            False
                        ])

                        flag_new_object = False
                        # draw bounding new box obj
                        text_new_object = "New " + name + ' : ' + "{:.2f}".format(conf) + ' %'
                        draw_box_obj(color_detection, obj_center_x, obj_center_y, xmin, xmax, ymin, ymax, text_new_object, full_frame)

            # print("TRACKLIST::::::: ", tracking_list_obj)
            if(tracking_list_obj is not None):
                if(len(tracking_list_obj) > 1):
                    # if objective not move so delete degress life one point
                    for i, track_obj in enumerate(tracking_list_obj):
                        # print("TRACK LIST", track_obj[0])
                        if track_obj[2] is False:
                            if(track_obj[1] <= 0):
                                tracking_list_delete_obj.append(i)
                            else:
                                tracking_list_obj[i][1] -= 1
                                tracking_list_obj[i][2] = True 
                    
                    if(len(tracking_list_delete_obj) > 0):
                        # print("LIST DELETE", tracking_list_delete_obj)
                        for i, list_del in reversed(list(enumerate(tracking_list_delete_obj))):
                            # print(i, "index delete : " ,list_del)
                            tracking_list_obj.pop(list_del)
                        # reset list delete tracking obj
                        tracking_list_delete_obj = []

        fps = 1./(time.time()-time_start)
        if(fps_avg == 0):
            fps_avg = fps
        else:
            fps_avg = (fps_avg + fps) // 2
        cv.putText(full_frame, "FPS: {:.2f}".format(fps), (5,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
        cv.putText(full_frame, "COUNT VEHECLE: " + str(count_car), (5,40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
        cv.putText(full_frame, "COUNT ACCIDENT: " + str(count_accident), (5,60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
        cv.putText(full_frame, "SPEED AVERAGE FLOW: {:.2f}".format(speed_average_flow), (5,80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
        

        # datetime object containing current date and time
        # now = datetime.now()
        # dd/mm/YY H:M:S
        # dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        # print("date and time =", dt_string)
        # cv.putText(full_frame, "date and time: " + str(dt_string), (5,90), cv.FONT_HERSHEY_SIMPLEX, 0.50, (255,0,255), 2)
        
        if cfg.MAIN.USE_SOCKET:
            res, frame = cv.imencode('.jpg', full_frame)    # from image to binary buffer
            data = base64.b64encode(frame)              # convert to base64 format
            sio.emit('imageid{}'.format(cfg.MAIN.SOCKET_ID_NAME), data)       # send to socket           
        
        if cfg.MAIN.SHOW_VID:
            cv.imshow('Full frame : {}'.format(cfg.MAIN.INFLUXDE_MEASURE_NAME), full_frame)
        # SAVE VIDEO
        if cfg.MAIN.SAVE_VID:
            video.write(full_frame)

        # cv.imwrite(PATH_SAVE_IMAGE_TO_WEB, full_frame)

        # ## send data to socket
        # name_to_web = 'cctv01'
        # send_obj = ['cctv01', full_frame]
        # sender.send_pyobj(send_obj)
        count_time_frame += 1
        # check time to get Data and reset
        time_check_get_speed_flow = cfg.MAIN.TIME_TO_GET_DATA * cfg.MAIN.FRAME_RATE_SOURCE * 60
        if(count_time_frame >= time_check_get_speed_flow):
            count_time_frame = 0
            process_time_minute += 1
            if (process_time_minute < 10 and process_time_hours <= 0):
                process_time_string = 'day{}-00.0{}'.format(process_time_day, process_time_minute)
            elif (process_time_minute >= 10 and process_time_minute < 60 and process_time_hours <= 0):
                process_time_string = 'day{}-00.{}'.format(process_time_day, process_time_minute)
            elif(process_time_minute == 60 and process_time_hours <= 0):
                process_time_hours += 1
                process_time_minute = 0
                process_time_string = 'day{}-0{}.0{}'.format(process_time_day, process_time_hours, process_time_minute)
            elif(process_time_minute < 10 and process_time_hours > 0 and process_time_hours < 10):
                process_time_string = 'day{}-0{}.0{}'.format(process_time_day, process_time_hours, process_time_minute)
            elif(process_time_minute >= 10 and process_time_minute < 60 and process_time_hours > 0 and process_time_hours < 10):
                process_time_string = 'day{}-0{}.{}'.format(process_time_day, process_time_hours, process_time_minute)
            elif(process_time_minute == 60 and process_time_hours > 0 and process_time_hours < 10):
                process_time_hours += 1
                process_time_minute = 0
                process_time_string = 'day{}-0{}.0{}'.format(process_time_day, process_time_hours, process_time_minute)
            elif(process_time_minute < 10 and process_time_hours >= 10 and process_time_hours < 60):
                process_time_string = 'day{}-{}.0{}'.format(process_time_day, process_time_hours, process_time_minute)
            elif(process_time_minute >= 10 and process_time_minute < 60 and process_time_hours >= 10 and process_time_hours < 60):
                process_time_string = 'day{}-{}.{}'.format(process_time_day, process_time_hours, process_time_minute)
            elif(process_time_minute == 60 and process_time_hours >= 10 and process_time_hours < 60):
                process_time_hours += 1
                process_time_minute = 0
                process_time_string = 'day{}-{}.0{}'.format(process_time_day, process_time_hours, process_time_minute)
            elif(process_time_hours == 60):
                process_time_day += 1
                process_time_hours = 0
                process_time_string = 'day{}-{}.0{}'.format(process_time_day, process_time_hours, process_time_minute)
  
            # datetime object containing current date and time
            now = datetime.now()
            # print("now =", now)
            # dd/mm/YY H:M:S
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            print("date and time =", dt_string)
            
            
            data = [dt_string, process_time_string, fps, speed_average_flow, count_car, count_accident, count_vehicle_all]

            if cfg.MAIN.SAVE_CSV:
                with open(cfg.MAIN.PATH_SAVE_CSV, 'a', encoding='utf8', newline='') as f:
                    writer = csv.writer(f)

                    if(check_header == 0):
                        # write the header
                        writer.writerow(header)
                        check_header = 1
                    # write the data
                    writer.writerow(data)
                    
                
            if cfg.MAIN.USE_INFLUXDB:
                print('use influxdb measurement : {}'.format(cfg.MAIN.INFLUXDE_MEASURE_NAME))   
                
                #  setup database
                client = InfluxDBClient(host=cfg.MAIN.INFLUXDB_HOST_IP, port=cfg.MAIN.INFLUXDB_PORT, username=cfg.MAIN.INFLUXDB_USERNAME, password=cfg.MAIN.INFLUXDB_PASSWORD, database=cfg.MAIN.INFLUXDB_DATABASE)
                client.create_database(cfg.MAIN.INFLUXDB_DATABASE)
                client.get_list_database()
                client.switch_database(cfg.MAIN.INFLUXDB_DATABASE)
                influxdb_data = {
                    "measurement": cfg.MAIN.INFLUXDE_MEASURE_NAME,
                    "fields": {
                        "vehicle_count": count_car,
                        "speed_estimate": speed_average_flow,
                        "accident_estimate": count_accident
                    }
                }
                influxdb_json_payload = []
                influxdb_json_payload.append(influxdb_data)
                client.write_points(influxdb_json_payload)
                
                
            # reset parameter
            count_car = 0
            count_accident = 0
            speed_average_flow = 0
        # frame_length += 1
        if(cv.waitKey(1) & 0xFF==ord('q')):
            break


    print("COUNT VEHECLE : ",count_car)
    print("ACCIDENT : ",count_accident)
    print("SPEED AVAGE FLOW : ", speed_average_flow)
    print("COUNT_VEHECLE_ALL : ", count_vehicle_all)
    now = datetime.now()
    # print("now =", now)
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    data = [dt_string, process_time_string, fps, speed_average_flow, count_car, count_accident, count_vehicle_all]
    if cfg.MAIN.SAVE_CSV:
        with open(cfg.MAIN.PATH_SAVE_CSV, 'a', encoding='utf8', newline='') as f:
            writer = csv.writer(f)

            if(check_header == 0):
                # write the header
                writer.writerow(header)
                check_header = 1
            # write the data
            writer.writerow(data)
                
    cap.release()
    cv.destroyAllWindows()
    os._exit(0) 
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=ROOT / 'configs/manage1.yaml', help='config path(s)')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)