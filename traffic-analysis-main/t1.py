from os import name
import torch
import cv2 as cv
import time
import numpy as np
import math

# import zmq

# ## For Web 
# context = zmq.Context()
# # send messages to socket
# sender = context.socket(zmq.PUSH)
# sender.bind("tcp://*:5557")



torch.cuda.is_available()

############ CONFIGURE PARAMETERS #################################
# MODEL MANAGEMENT
INPUT_FILE = "video/test02.mp4"
# INPUT_FILE = "video/cars.mp4"

INPUT_MODEL = "config_model/yolov5m.pt"
CONFIDENCE_THRESHOLD = 0.4

# FLAGE EVENT
flag_config_ROI = False

# COLOR OBJECTIVES
color_car_detection = (0, 255, 0)
color_car_tracking = (255, 255, 0)

color_truck_detection = (0, 0, 255)
color_truck_tracking = (127, 127, 127)

color_bus_detection = (255, 0, 0)
color_bus_tracking = (255, 127, 127)

color_motorcycle_detection = (255, 0, 0)
color_motorcycle_tracking = (255, 127, 127)


## TRACKING PARAMETERS
# list parameter definition
# tracking_list_obj = [
#       (obj_center_x, obj_center_y), 
#       life for delete tracking, 
#       flag for decrease life, 
#       number for count car,
#       (number for start calculate speed, distance stack, speed is pixels(or kilimeters) per hours)
# ] 
tracking_list_obj = None  
tracking_list_delete_obj = []

# configure tracking
LIFE = 30
# tracking_x_distance = 30
# tracking_y_distance = 30
TRACKING_LIMIT_DISTANCE = 50

# configure for count car if car tracked x round
FRAME_TO_COUNT = 8

# configure for calculate speed car
FRAME_TO_CALCULATE_SPEED = 10
SPEED_OUT_OF_SIGHT  = 30 # Km/hr
SPEED_OUT_OF_SIGHT_CAR_STOP = 9 # Km/hr

# meter per pixels need to calculate from standart road with camera  meter/pixels
# METERS_PER_PIXELS  = 0.04136914426
METERS_PER_PIXELS  = 0.03
# METERS_PER_PIXELS  = 1/8.8
# METERS_PER_PIXELS = 3/35

# METERS_PER_PIXELS = 0.25
FRAME_RATE_SOURCE = 30



# OUTPUTS
count_car = 0
count_accident = 0


###################################################################

############ FUNCTIONAL ############################################
# LIST TYPE OBJECT DETECTION
def list_obj_detection(name):
    if(name == 'car' or name == 'bus' or name == 'motorcycle'):
        return True
    else:
        return False

# SELECT COLOR FOR OBJECTIVE
def select_color_object(name):
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

# DRAWING BOUNDING BOXES
def draw_box_obj(color, center_x, center_y, d_xmin, d_xmax, d_ymin, d_ymax, text ):
    cv.circle(frame,(center_x, center_y), 1, (0,0,255), 2)
    cv.rectangle(frame, (d_xmin, d_ymin), (d_xmax, d_ymax), color, 2)
    cv.putText(frame, text, (d_xmin, d_ymin-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (152, 255, 50), 2)

# UPDATE POSITION LIFE POINT OF COUNTING
def update_tracking_obj(index, center_x, center_y, distance):
        tracking_list_obj[index][0][0] = center_x
        tracking_list_obj[index][0][1] = center_y
        tracking_list_obj[index][1] = LIFE
        tracking_list_obj[index][2] = True
        tracking_list_obj[index][3] += 1
        tracking_list_obj[index][4][0] += 1
        tracking_list_obj[index][4][1] += distance


# UPDATE SPEED OBJECTIVE AND RESET DISTANCE STACK
def update_speed_object(index, speed):
    tracking_list_obj[index][4][1] = 0
    tracking_list_obj[index][4][2] = speed


# CALCULATE DISTANCE OBJECTIVE
def calculate_distance(x1, y1, x2, y2):
    # return math.sqrt(((abs(x2 - x1))**2) + ((abs(y2 - y1))**2))
    distance = math.sqrt(((abs(x2 - x1))**2) + ((abs(y2 - y1))**2))
    return distance
    

# CALCULATE SPEED IN PIXEL / MINUS UNIT OBJECTIVE
def calculate_speed(d_pixel, fps_sp):

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
    speed_per_meters_km_hr  = d_pixel * METERS_PER_PIXELS * fps_sp * 3.6
    return speed_per_meters_km_hr


# CALCULATE AVERAGE SPEED OBJECTIVE
def estimate_speed(spd, frame_to_estimate):
    speed = spd / frame_to_estimate
    return speed

def estimate_speed2(index, d_pixel, fps_sp, frame_to_cal):

    # pixels per seconds
    # speed_per_pixel_second  = (d_pixel/frame_to_cal) * fps_sp
    # return speed_per_pixel

    # pixels per hours
    # speed_per_pixel_hour  = (d_pixel/frame_to_cal) * fps_sp * 3.6
    # return speed_per_pixel_hour

    # meters per seconds
    # speed_per_meters_second  = (d_pixel/frame_to_cal) * METERS_PER_PIXELS * fps_sp
    # return speed_per_meters_second
    
    # kilometer per hour 
    speed_per_meters_km_hr  = (d_pixel/(frame_to_cal + 1)) * METERS_PER_PIXELS * fps_sp * 3.6
    if (tracking_list_obj[index][4][2] > 0 ): 
        if(speed_per_meters_km_hr <= SPEED_OUT_OF_SIGHT_CAR_STOP):
            speed_per_meters_km_hr = 0
        # else:
        estimate_cal = abs(speed_estimate_obj - tracking_list_obj[index][4][2])
        # if(estimate_cal > SPEED_OUT_OF_SIGHT ):
        #     speed_per_meters_km_hr = tracking_list_obj[index][4][2]


    return speed_per_meters_km_hr
    

    

###################################################################


# RUN MODEL 
model = torch.hub.load('C:/project/traffic-analysis', 'custom', path=INPUT_MODEL, source='local') 
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
# 
cap = cv.VideoCapture(INPUT_FILE)
ret, image_origin = cap.read()
image = cv.resize(image_origin, (960, 540))  

# SAVE VIDEO CONFIG
PATH_SAVE = 'outputs/t5.mp4'
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
    roi_xmin = 131
    roi_ymin = 152
    roi_xmax = 789
    roi_ymax = 530

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
    
    results = model(frame, size=640, classes=[2, 3])  # includes NMS
    
    
    # Results
    # results.print()
    # print(results.pandas().xyxy[0])  # img1 predictions (pandas)

    for idx, obj in results.pandas().xyxy[0].iterrows():

        if(obj['confidence'] > 0.3 and list_obj_detection(obj['name']) ):
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
                tracking_list_obj.append([[obj_center_x, obj_center_y], LIFE, True, 0, [0, 0, 0]])

                # draw bounding new box obj
                text_new_object = "New " + name + ' : ' + "{:.2f}".format(conf) + ' %'
                draw_box_obj(color_detection, obj_center_x, obj_center_y, xmin, xmax, ymin, ymax, text_new_object)
            else:
                # check id old object ?
                for i, track_obj in enumerate(tracking_list_obj):
                    speed_obj = 0
                    distance_obj = 0
                    # print("track objective", track_obj)

                    # calculate distance per frame
                    distance_obj = calculate_distance(obj_center_x, obj_center_y, track_obj[0][0], track_obj[0][1])
                    # print("DISTANCE BEETWEEN 2 POINT: ", distance_obj)

                    # if(abs(obj_center_x - track_obj[0][0]) < tracking_x_distance and abs(obj_center_y - track_obj[0][1]) < tracking_y_distance):
                    if(distance_obj <= TRACKING_LIMIT_DISTANCE):

                        # calculate speed in one frame
                        # speed_obj = calculate_speed(distance_obj, FRAME_RATE_SOURCE)

                        # update position life countCondition and distance objective tracked
                        update_tracking_obj(i, obj_center_x, obj_center_y, distance_obj)

                        # condition to counting object
                        if (track_obj[3] == FRAME_TO_COUNT):
                            count_car += 1
                            
                        
                        # condition to estimate speed object
                        if((track_obj[4][0] % FRAME_TO_CALCULATE_SPEED) == 0):
                            speed_estimate_obj = estimate_speed2(i, track_obj[4][1], FRAME_RATE_SOURCE, FRAME_TO_CALCULATE_SPEED)
                            update_speed_object(i, speed_estimate_obj)

                        # draw bounding box for tracking object
                        text_tracking_object = "Tracking " + name + ' : ' + "{:.2f}".format(conf) + ' %' + " {:.2f}".format(track_obj[4][2]) + ' km/hr'
                        draw_box_obj(color_tracking, obj_center_x, obj_center_y, xmin, xmax, ymin, ymax, text_tracking_object)
                        flag_new_object = False
                        break
                    else:
                        flag_new_object = True
                        tracking_list_obj[i][2] = False
                
                if flag_new_object is True:
                    # new objective so adding list tracking
                    tracking_list_obj.append([[obj_center_x, obj_center_y], LIFE, True, 0, [0, 0, 0]])
                    flag_new_object = False
                    # draw bounding new box obj
                    text_new_object = "New " + name + ' : ' + "{:.2f}".format(conf) + ' %'
                    draw_box_obj(color_detection, obj_center_x, obj_center_y, xmin, xmax, ymin, ymax, text_new_object)

        # print("TRACKLIST::::::: ", tracking_list_obj)
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
    cv.putText(full_frame, "FPS: {:.2f}".format(fps), (5,30), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,255), 2)
    cv.putText(full_frame, "COUNT: " + str(count_car), (5,60), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,255), 2)
    cv.imshow('Full frame', full_frame)

    # SAVE VIDEO
    video.write(full_frame)

    cv.imwrite(PATH_SAVE_IMAGE_TO_WEB, full_frame)

    # ## send data to socket
    # name_to_web = 'cctv01'
    # send_obj = ['cctv01', full_frame]
    # sender.send_pyobj(send_obj)

    # frame_length += 1
    if(cv.waitKey(1) & 0xFF==ord('q')):
       break


print("COUNT CAR : ",count_car)
cap.release()
cv.destroyAllWindows()
