import logging
import time
import cv2
import numpy as np
#from djitellopy import tello
from time import sleep
import cv2 as cv 

CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    global count
    x=0
    y=0
    w=0
    h=0
    # creating empty list to add objects data
    data_list =[]
    for (classid, score, box) in zip(classes, scores, boxes):
        if classid == 0:
            label = "%s : %f" % (class_names[classid], score)
        # draw rectangle on and label on objec
            #print("box=",box)
            x=box[0]
            y=box[1]
            w=box[2]
            h=box[3]
            cx = x + w // 2
            cy = y + h // 2
            area = w * h
            #print("area",area)
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            for i in range (80):
                if classid == i: # person class id 
                    data_list.append([class_names[classid], box[2], (box[0], box[1]-2),(x,y,w,h)])
    return data_list,image

#new-add
def get_each_frame_info(on_off):
    check = 0
    global data
    #global data_arr
    for d in data:
        if d[0] =='person':
            check=1
            x, y = d[2]
            x1,y1,w1,h1=d[3]
            data_arr.append([count_2,x1,y1,w1,h1])

    if check == 0:
        data_arr.append([count_2,0,0,0,0])
    abc = data_arr.copy()
    print("0000000=", data_arr)
    data_arr.clear()

# Camera Setting
Camera_Width = 720
Camera_Height = 480

#無人機初始化,開始時升高到一定的程度
#mytello = tello.Tello()
#mytello.connect() #wifi連線
#mytello.streamon() #開啓視訊鏡頭
#mytello.takeoff() #起飛
#print("電池格數剩: {}%".format(mytello.get_battery()))
#mytello.send_rc_control(0, 0, 25, 0) #上升至一定的高度
sleep(2)

w, h = 360, 240
count=0
count_2=0
#check=0
data_arr = []
cap = cv2.VideoCapture('123.mp4')
while True:
    ret, OriginalImage = cap.read()
    Image = cv2.resize(OriginalImage, (Camera_Width, Camera_Height))
    frame=Image
    Img_Name = "img_frame/" + str(count_2)+ ".jpg"
    count_2=count_2+1
    cv2.imwrite(Img_Name, frame)

    data,img = object_detector(frame)
    #print(data)
    get_each_frame_info(on_off)
    '''
    check=0
    for d in data:
        if d[0] =='person':
            check=1
            x, y = d[2]
            x1,y1,w1,h1=d[3]
            data_arr.append([count_2,x1,y1,w1,h1])

    if check == 0:
        data_arr.append([count_2,0,0,0,0])
    check = 0

    abc=data_arr.copy()
    print("0000000=",data_arr)
    data_arr.clear()
    #print(frame)
    '''