# -*- coding: UTF-8 -*-
import tkinter as tk
from PIL import ImageTk, Image, ImageDraw, ImageFont
import cv2
import os
import shutil
import numpy as np
#from djitellopy import tello

import pandas as pd
import queue
import math
from cmp_person_similarity.predict_from_dataset import similarity_detect

from datetime import datetime
###initial used

#set
frame_size_w = 720
frame_size_h = 480
clear_frame_img = 0
temp_image_amount = 5
img_video = './video/123.mp4'  #temporarily video
fade_color = ["#FF0000","#FF8800","#FFFF00","#00FFCC","#0000FF","#7700FF","#FF00FF"]

#global_variable-used
blocks_info = []
selected_block_num = -1
pre_frame_selected_block = []
area_list = []
redetect_flag = 0
pre_center_point_x = 0
pre_center_point_y = 0
pre_avg_moment = 0#for freak reblock use,detect range will get very large
point_movements = queue.Queue(maxsize=10)#only put 10 frame info
avg_movement = 0
loss_signal = -1
frame_count =0
activate_cmp_detector=0
find_times = 0
high_acc_index =0
search_prob_list = []
dis_list = []
last_point_x = 0
last_point_y = 0
last_bbox = 0
quick_refind_flag = 0
range_search_flag = 0
area_decrease= 0
#============================#
#   create folder
#============================#
#create temp file to compare person image
if not os.path.exists('./temp'):
    os.mkdir('./temp')

#clear temp data
else:
    shutil.rmtree('./temp')
    os.mkdir('./temp')

#save each frame video frames
if not os.path.exists('./video_frames'):
    os.mkdir('./video_frames')

#clear video_frames data
else:
    shutil.rmtree('./video_frames')
    os.mkdir('./video_frames')

#create history file to save record
if not os.path.exists('./history'):
    os.mkdir('./history')

#create video_frame to recheck
if not os.path.exists('./video_frames'):
    os.mkdir('./video_frames')

#create a new tracking data for a person
history_len = len(os.listdir('./history'))
os.mkdir('./history'+'/'+str(history_len))

def restart():
    global blocks_info
    global selected_block_num
    global pre_frame_selected_block
    global redetect_flag
    global pre_all_area
    global pre_center_point_x
    global pre_center_point_y
    global point_movements   # only put 10 frame info
    global avg_movement
    global loss_signal
    global frame_count
    global activate_cmp_detector
    global find_times
    global high_acc_index
    global search_prob_list
    global dis_list
    global last_point_x
    global last_point_y
    global last_bbox
    global quick_refind_flag
    global range_search_flag
    global area_decrease
    # global_variable-used
    blocks_info = []
    selected_block_num = -1
    pre_frame_selected_block = []
    redetect_flag = 0
    pre_all_area = 0
    pre_center_point_x = 0
    pre_center_point_y = 0
    point_movements = queue.Queue(maxsize=10)  # only put 10 frame info
    avg_movement = 0
    loss_signal = -1
    frame_count = 0
    activate_cmp_detector = 0
    find_times = 0
    high_acc_index = 0
    search_prob_list = []
    dis_list = []
    last_point_x = 0
    last_point_y = 0
    last_bbox = 0
    quick_refind_flag = 0
    range_search_flag = 0
    area_decrease= 0



#####yolov4 detector
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3
class_names = []
with open("yoloIOT/classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

#  setttng up opencv net
yoloNet = cv2.dnn.readNet('yoloIOT/yolov4-tiny.weights', 'yoloIOT/yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)


def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    global count
    x = 0
    y = 0
    w = 0
    h = 0
    # creating empty list to add objects data
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        if classid == 0:
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            cx = x + w // 2
            cy = y + h // 2
            area = w * h
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            for i in range(80):
                if classid == i:  # person class id
                    data_list.append([class_names[classid], box[2], (box[0], box[1] - 2), (x, y, w, h)])
    return data_list, image


# Camera Setting
#Camera_Width = 720
#Camera_Height = 480

# 無人機初始化,開始時升高到一定的程度
# mytello = tello.Tello()
# mytello.connect() #wifi連線
# mytello.streamon() #開啓視訊鏡頭
# mytello.takeoff() #起飛
# print("電池格數剩: {}%".format(mytello.get_battery()))
# mytello.send_rc_control(0, 0, 25, 0) #上升至一定的高度
# sleep(2)

w, h = frame_size_w, frame_size_h
count = 0
count_2 = 0
check = 0
data_arr = []


#============================#
#   tracking algorithm
#============================#
def track_num_selected(mouse_x,mouse_y):
    global blocks_info
    global selected_block_num
    global pre_frame_selected_block
    global redetect_flag
    global area_list
    global pre_center_point_x
    global pre_center_point_y
    for block_num in range(len(blocks_info)):
        if blocks_info[block_num][0]<mouse_x<(blocks_info[block_num][0]+blocks_info[block_num][2]) and blocks_info[block_num][1]<mouse_y<(blocks_info[block_num][1]+blocks_info[block_num][3]):
            selected_block_num = block_num
            pre_frame_selected_block = blocks_info[selected_block_num]
            redetect_flag = 1
            area_list.append(blocks_info[selected_block_num][2] * blocks_info[selected_block_num][3])
            pre_center_point_x = blocks_info[selected_block_num][0]+(blocks_info[selected_block_num][2]/2)
            pre_center_point_y = blocks_info[selected_block_num][1]+(blocks_info[selected_block_num][3]/2)

def get_iou(bbox_ai, bbox_gt):
    iou_x = max(bbox_ai[0], bbox_gt[0]) # x
    iou_y = max(bbox_ai[1], bbox_gt[1]) # y
    iou_w = min(bbox_ai[2]+bbox_ai[0], bbox_gt[2]+bbox_gt[0]) - iou_x # w
    iou_w = max(iou_w, 0)
    #print(f'{iou_w=}')
    iou_h = min(bbox_ai[3]+bbox_ai[1], bbox_gt[3]+bbox_gt[1]) - iou_y # h
    iou_h = max(iou_h, 0)
    #print(f'{iou_h=}')
    iou_area = iou_w * iou_h
    #print(f'{iou_area=}')
    all_area = bbox_ai[2]*bbox_ai[3] + bbox_gt[2]*bbox_gt[3] - iou_area
    #print(f'{all_area=}')
    return max(iou_area/all_area , 0)

def tracking():
    global pre_frame_selected_block
    global selected_block_num
    global blocks_info
    #global pre_all_area
    global pre_center_point_x
    global pre_center_point_y
    global loss_signal
    global frame_count
    global point_movements
    global avg_movement
    global activate_cmp_detector
    global find_times
    global high_acc_index
    global search_prob_list
    global dis_list
    global last_point_x
    global last_point_y
    global last_bbox
    global quick_refind_flag
    global range_search_flag
    global area_decrease
    print("#######################frame"+str(frame_count))
    print("detect_blocks:",blocks_info)
    index_count = 0
    temp_iou = []
    temp_index_max2min = []
    print("pre_frame_selected_block: ",pre_frame_selected_block)
    print("loss_signal: ", loss_signal)
    print("pre_center_point_x,pre_center_point_y :",pre_center_point_x,pre_center_point_y)
    print("activate_cmp_detector :",activate_cmp_detector)
    print("area_list:",area_list)


    #area relation
    if len(area_list) > 10:
        area_list.pop(0)

    person_out_of_block = 0
    #check the area gradually decent or not,if true the person walk out the camera block area
    for area_num in range(len(area_list)-1):
        if area_list[area_num] > area_list[area_num+1]:
            person_out_of_block += 1

    if person_out_of_block == 7:#almost decrease
        area_decrease = 1
    else:
        area_decrease = 0




    if loss_signal >= 0:
        pre_center_point_x = last_point_x
        pre_center_point_y = last_point_y
        pre_frame_selected_block = last_bbox

    for j in range(len(blocks_info)):
        result = get_iou(pre_frame_selected_block, blocks_info[j])
        temp_iou.append(result)
    print("all person block cmp pre block iou:",temp_iou)
    sorted_list = sorted(temp_iou,reverse = True)#max to min
    temp_index_max2min = [temp_iou.index(i) for i in sorted_list]
    print("sort index: ",temp_index_max2min)

    index = temp_index_max2min[index_count]
    #print("save_area:", frame_count%5)
    if frame_count%5 == 0:
        area_list.append(blocks_info[index][2]*blocks_info[index][3])



    if activate_cmp_detector == 1:
        refind_temp = []
        search_prob_list.clear()
        #pre_frame_selected_block = last_bbox
        print("xxxxxxxxxxxxx start cmp detector search network")
        print("find_time:",find_times)
        print(loss_signal)
        for g in range(len(blocks_info)):
            search_peron_img = clear_frame_img.crop(((
            blocks_info[g][0], blocks_info[g][1],
            (blocks_info[g][0] + blocks_info[g][2]),
            (blocks_info[g][1] + blocks_info[g][3]))))
            prob_cmp_result = similarity_detect(search_peron_img)[0][0]#get value deparcet blank
            refind_temp.append(prob_cmp_result)
        print("refind_temp_list:",refind_temp)
        #high_acc_index = np.argmax(np.array(refind_temp))
        sorted_refind_temp = sorted(refind_temp, reverse=True)  # max to min
        high_acc_cmp_index_list = [refind_temp.index(i) for i in sorted_refind_temp]
        print(high_acc_cmp_index_list)

        #@@quick_refind
        if loss_signal <= 3:
            print("=====start quick reblock algorithm======")
            #check all list only one get high iou that's the loc to reblock
            if temp_iou[temp_index_max2min[1]] == 0.0 and temp_iou[temp_index_max2min[0]] > 0.3:
                selected_block_num = temp_index_max2min[0]
                pre_frame_selected_block = blocks_info[selected_block_num]
                loss_signal = -1
                activate_cmp_detector = 0
                quick_refind_flag = 1

            else:
                for q_refind_index in temp_index_max2min:
                    if temp_iou[q_refind_index] > 0.7 and refind_temp[q_refind_index] > 0.97:
                        selected_block_num = q_refind_index
                        pre_frame_selected_block = blocks_info[selected_block_num]
                        loss_signal = -1
                        activate_cmp_detector = 0
                        quick_refind_flag = 1

        #@@refind
        else:
            print("===== start range search algorithm ======")
            #confirm mode
            for k in range(len(blocks_info)):
                guess_center_point_x = blocks_info[k][0] + blocks_info[k][2] / 2
                guess_center_point_y = blocks_info[k][1] + blocks_info[k][3] / 2
                dis = math.sqrt((last_point_x - guess_center_point_x) ** 2 + (last_point_y - guess_center_point_y) ** 2)
                dis_list.append(dis)
            print("dist_list: ", dis_list)
            for guess in range(len(high_acc_cmp_index_list)):
                if refind_temp[guess] > 0.1:
                    search_prob_list.append(high_acc_cmp_index_list[guess])
            print("search_prob_list",search_prob_list)
            #check
            check_flag = 0
            for p in search_prob_list:
                if dis_list[p]<avg_movement*3 and refind_temp[k] > 0.90:
                    check_flag += 1
                    temp_index = p
                    break
            if check_flag > 1 :
                find_times += 1
            else:
                find_times = 0

            if find_times == 5:
                range_search_flag = 1
                activate_cmp_detector = 0
                selected_block_num = temp_index
                pre_frame_selected_block = blocks_info[selected_block_num]

        if quick_refind_flag == 0 and range_search_flag == 0 and blocks_info[0] != [0, 0, 0, 0]:
            loss_signal+=1

        print("avg_speed_movment: ",avg_movement)
        print("loss_signal : ",loss_signal)

    #no info not update the block
    if blocks_info[0] == [0, 0, 0, 0]:
        print("no block info not update tracking")
        loss_signal +=1

    #fast reblock
    elif quick_refind_flag == 1 and activate_cmp_detector == 0:
        find_times = 0
        loss_signal = -1
        quick_refind_flag = 0
        print("reblock successfull!!!")

    elif range_search_flag == 1 and activate_cmp_detector ==0:
        find_times = 0
        loss_signal = -1
        range_search_flag = 0

    elif activate_cmp_detector == 0:
        # center point movement
        center_point_x = blocks_info[index][0] + (blocks_info[index][2] / 2)
        center_point_y = blocks_info[index][1] + (blocks_info[index][3] / 2)
        dis = math.sqrt((pre_center_point_x - center_point_x) ** 2 + (pre_center_point_y - center_point_y) ** 2)
        # GET ONLY LEAST 10 NEW DATA
        print("@@@@@point_movement>>")
        print("distance:",dis)
        print("avg_monements:",avg_movement)
        if point_movements.full():
            print("|set 10 movements full| start operate the movement detect")
            avg_movement = sum(list(point_movements.queue)) / 10
            print("avg_movement: ",avg_movement)
            print("frame_dis: ",dis)
            point_movements.get()

        if dis > avg_movement*2 and temp_iou[index]<0.3 and avg_movement!=0: #or avg_movement*0.5 < dis:
            print("warning: overlay_block_error_encounter or loss !!!!!")
            area_list.pop(-1)
            activate_cmp_detector = 1
            last_point_x = pre_center_point_x
            last_point_y = pre_center_point_y
            last_bbox = pre_frame_selected_block
            loss_signal = 0
            point_movements.queue.clear()

        elif dis != 0 and avg_movement != 0 and dis > avg_movement*10 and area_decrease==1:#it will happen when reblock freaky happened
            print("the block moving really strange or loss")
            activate_cmp_detector = 1
            last_point_x = pre_center_point_x
            last_point_y = pre_center_point_y
            last_bbox = pre_frame_selected_block
            loss_signal = 10#direct jump to stage 2 reblock


        point_movements.put(dis)
        pre_center_point_x = center_point_x
        pre_center_point_y = center_point_y

        # update index block
        selected_block_num = index
        pre_frame_selected_block = blocks_info[index]
    temp_iou.clear()
    dis_list.clear()

#============================#
#   Image process
#============================#
#for pil image concate used
def get_concat_h_cut(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, min(im1.height, im2.height)))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def write_text(im1,text,loc,size):
    text = text
    draw = ImageDraw.Draw(im1)
    font = ImageFont.truetype("arial.ttf", size=size)
    draw.text(loc, text, fill="black", font=font, align="center")
    return im1

#============================#
#   GUI
#============================#
capture = cv2.VideoCapture(0)  # 開啟相機，0為預設筆電內建相機
capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # 設置影像參數
#capture.set(3, 720)  # 像素
#capture.set(4, 480)  # 像素
#capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
#capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def check():
    restart()
    global capture
    video_flag =1
    #if capture.isOpened():  # 判斷相機是否有開啟
    if capture.isOpened() and video_flag == 0:  # 判斷相機是否有開啟
        open()
    else:
        #capture = cv2.VideoCapture(0)
        capture = cv2.VideoCapture(img_video)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # 設置影像參數
        capture.set(3, 720)  # 像素
        capture.set(4, 480)  # 像素
        open()

def block_data(frame_tag):
    global block_info
    data = pd.read_csv('./pd_data.csv')
    block_info.append([data.iloc[frame_tag, 2],data.iloc[frame_tag, 3],data.iloc[frame_tag, 4],data.iloc[frame_tag, 5]])
    return block_info

#open folder
def open_folder():
    #for windows
    history_folder = r'C:\Users\Alex\PycharmProjects\iot_tracker\history'
    #for linux
    #history_folder = './history'??? not test
    os.startfile(history_folder)

#mouse click selected person
def pre_selected_person(event):
    #print("Mouse pre_selected_person corrdinate" + str(event.x) + "," + str(event.y))
    mouse_x = event.x
    mouse_y = event.y
    if 0 < mouse_x < frame_size_w and 0 < mouse_y < frame_size_h:
        track_num_selected(mouse_x,mouse_y)

def cancel_pre_selected_person(event):
    global selected_block_num
    global redetect_flag
    #print("Mouse pre_selected_person corrdinate" + str(event.x) + "," + str(event.y))
    mouse_x = event.x
    mouse_y = event.y
    if 0 < mouse_x < frame_size_w and 0 < mouse_y < frame_size_h:
        selected_block_num = -1
        redetect_flag = 0

def open():
    global s
    global blocks_info
    global frame_count
    global clear_frame_img
    frame_count += 1
    ret, frame = capture.read()  #get camera image
    frame = cv2.resize(frame, (frame_size_w, frame_size_h), interpolation=cv2.INTER_AREA)
    clear_frame_img = frame.copy()
    #cv2.imwrite('img_video.jpg',frame)  # 儲存圖片
    data, img = object_detector(frame)
    check = 0
    for d in data:
        if d[0] == 'person':
            check = 1
            x, y = d[2]
            x1, y1, w1, h1 = d[3]
            data_arr.append([x1, y1, w1, h1])
    if check == 0:
        data_arr.append([0, 0, 0, 0])
        check = 0

    blocks_info = data_arr.copy()
    data_arr.clear()

    #select person mouse event
    win.bind("<Button-1>", pre_selected_person)  # right key
    win.bind("<Button-3>", cancel_pre_selected_person)

    #draw block on Image
    #frame_img = Image.open('img_video.jpg')
    frame_img = Image.fromarray(clear_frame_img)
    clear_frame_img = frame_img.copy()

    for x in range(len(blocks_info)):
        draw = ImageDraw.Draw(frame_img)
        draw.rectangle(((blocks_info[x][0], blocks_info[x][1]), (blocks_info[x][0]+blocks_info[x][2], blocks_info[x][1]+blocks_info[x][3])), outline="green",width=3)#fill="green")

    if redetect_flag == 1:
            tracking()

    if loss_signal == -1:
        if selected_block_num != -1 and activate_cmp_detector == 0:
            # creating new Image object and past the mask on the display image
            mask = Image.new("RGB", (blocks_info[selected_block_num][2],blocks_info[selected_block_num][3]),color = (153, 153, 255))
            selected_person_img = frame_img.crop(((blocks_info[selected_block_num][0], blocks_info[selected_block_num][1], (blocks_info[selected_block_num][0]+blocks_info[selected_block_num][2]), (blocks_info[selected_block_num][1]+blocks_info[selected_block_num][3]))))
            blend_img = Image.blend(mask,selected_person_img,0.5)
            frame_img.paste(blend_img,(blocks_info[selected_block_num][0],blocks_info[selected_block_num][1]))


            #write tracking info image
            if frame_count%20 == 0 and redetect_flag == 1:
                save_img = clear_frame_img.copy()
                #now = datetime.now()
                current_time = datetime.now().strftime("%H:%M:%S")
                info_img = Image.new("RGB", (100, frame_size_h), color=(255, 255, 255))
                text  = "Date :"
                info_img = write_text(info_img, text, (10, 10), 20)  # img,text,loc,size

                text = datetime.now().strftime("%d/%m/%Y")
                info_img = write_text(info_img, text, (10, 40), 15)  # img,text,loc,size

                text = "Time: "
                info_img = write_text(info_img, text, (10, 80), 20)  # img,text,loc,size

                text = current_time
                info_img = write_text(info_img, text, (10, 120), 20)#img,text,loc,size

                save_img.paste(blend_img,(blocks_info[selected_block_num][0], blocks_info[selected_block_num][1]))
                vis = get_concat_h_cut(save_img,info_img)
                vis.save("./history"+"/"+str(history_len)+"/"+'track_info_'+str(frame_count)+'.jpg')

            #crop image to data base
            selected_person_img = clear_frame_img.crop(((
            blocks_info[selected_block_num][0], blocks_info[selected_block_num][1],
            (blocks_info[selected_block_num][0] + blocks_info[selected_block_num][2]),
            (blocks_info[selected_block_num][1] + blocks_info[selected_block_num][3]))))
            selected_person_img.save("./temp/"+str(frame_count).zfill(6)+"_frame_track_"+".jpg")
            if len(os.listdir('./temp'))>=temp_image_amount:
                os.remove("./temp/"+str(min(os.listdir('./temp'))))
                #print(0)

    # print lose stage on each display image
    if (0<=loss_signal <= 3):
        text = 'Quick reblock person........'
        draw = ImageDraw.Draw(frame_img)
        # drawing text size
        font = ImageFont.truetype("arial.ttf", size=20)
        draw.text((30, 30), text, fill="pink", font=font, align="center")

    if (3< loss_signal < 10):
        text = 'Range search........'
        draw = ImageDraw.Draw(frame_img)
        # drawing text size
        font = ImageFont.truetype("arial.ttf", size=20)
        draw.text((30, 30), text, fill="yellow", font=font, align="center")


    if (loss_signal >= 10):
        text = 'Lose tracking person........'
        draw = ImageDraw.Draw(frame_img)
        # drawing text size
        font = ImageFont.truetype("arial.ttf", size=20)
        draw.text((30, 30), text,fill= "red", font = font, align="center")


    if len(search_prob_list)>0: #and len(blocks_info):
        print("draw block_info",blocks_info)
        print("draw search_prob_list:",search_prob_list)
        for k in range(len(search_prob_list)):
            # draw pre_block
            if len(fade_color) > k:
                search_mask_color = fade_color[k]
            else:
                search_mask_color = fade_color[-1]
            mask = Image.new("RGB", (blocks_info[k][2], blocks_info[k][3]),
                             color=(search_mask_color))
            selected_person_img = frame_img.crop(((blocks_info[k][0], blocks_info[k][1],
                        (blocks_info[k][0] + blocks_info[k][2]),
                        (blocks_info[k][1] + blocks_info[k][3]))))
            blend_img = Image.blend(mask, selected_person_img, 0.5)  # img1*(1-0.5),img*0.5
            frame_img.paste(blend_img, (blocks_info[k][0], blocks_info[k][1]))

    #add frame num for debug
    text = 'frame'+str(frame_count)
    draw = ImageDraw.Draw(frame_img)
    # drawing text size
    font = ImageFont.truetype("arial.ttf", size=30)
    draw.text((0, 450), text,fill= "blue", font = font, align="center")
    frame_img.save("./video_frames"+"/"+str(frame_count)+".jpg")
    img_right = ImageTk.PhotoImage(frame_img) # 讀取圖片
    label_right.imgtk = img_right  # 換圖片
    label_right.config(image=img_right)  # 換圖片
    s = label_right.after(33, open)  # 持續執行open方法 #x ms refresh(1000/30=33.3)


def close():
    capture.release()  # 關閉相機
    label_right.after_cancel(s)  # 結束拍照
    label_right.config(image=img)  # 換圖片


# create window
win = tk.Tk()
win.title('person drone tracker GUI')
win.resizable(0,0)#can't resize gui window
win.geometry('900x500')
img = ImageTk.PhotoImage(Image.open('./instruction.jpg'))

# put image on label
label_right = tk.Label(win, height=frame_size_h, width=frame_size_w, bg='gray94', fg='blue', image=img)

# button
button_1 = tk.Button(win, text='start', bd=4, height=4, width=20, bg='gray94',command=check)
button_2 = tk.Button(win, text='person history', bd=4, height=4, width=20, bg='gray94',command=open_folder)
button_3 = tk.Button(win, text='exit', bd=4, height=4, width=20, bg='gray94',command=close)

# grid location
label_right.grid(row=1, column=0, padx=000, pady=000, sticky="nw")
button_1.grid(row=1, column=0, padx=730, pady=100, sticky="nw")
button_2.grid(row=1, column=0, padx=730, pady=200, sticky="nw")
button_3.grid(row=1, column=0, padx=730, pady=300, sticky="nw")
win.mainloop()  #execute window