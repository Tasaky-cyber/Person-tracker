# person_tracker_drone

:bulb: remember download the model first
> 1. get link in cmp_person_similarity/logs,and put download model in same location cmp_person_similarity/logs   
> 2. download tolov4-tiny.cft and yolov4-tiny.weights from https://github.com/Tianxiaomo/pytorch-YOLOv4 ,and put those two in yoloIOT  

execute main.py, work!!!

### flow
gui interfence >>> yolov4 >>> tracking algorithm >>> if loss, start refind (stage1:reblock,stage:long range search)  

### cmp_person_similarity  
compare two people by neural network feature extract  
1. perdict.py for (two picture compare)  
2. predict_from_dataset.py (input a picture compare the whole dataset,will get a avg similarity)  

### iot_tracker_gui
write in main file
