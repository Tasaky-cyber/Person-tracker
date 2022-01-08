# person_tracker
:bulb: remember download the model first
> 1. get link in cmp_person_similarity/logs,and put download model in same location cmp_person_similarity/logs   
> 2. download tolov4-tiny.cft and yolov4-tiny.weights from https://github.com/Tianxiaomo/pytorch-YOLOv4 ,and put those two in yoloIOT  
> 3. maybe encouter not found "arial.ttf" problem, you can search google to download  

!!! GUI - person history button can't work on linux, only on windows    

execute main.py, work!!!  
## Production
<p align="center">
  
https://user-images.githubusercontent.com/55420081/148530210-101bac02-8213-409e-987f-8fc721088c21.mp4 
  
</p>  
<p align="center">
  


https://user-images.githubusercontent.com/55420081/148595922-08a7f80a-59b3-401e-8f91-d9964abd307b.mp4


  
</p>  

## flow
gui interfence >>> yolov4 >>> tracking algorithm >>> if loss, start refind (stage1:reblock, stage2:long range search)>>if get retrack, else lose tracking  
![螢幕擷取畫面 2022-01-07 180032](https://user-images.githubusercontent.com/55420081/148527232-ce3b96b6-ad4c-41b5-ac94-307d6ec07968.png)  


## cmp_person_similarity  
compare two people by neural network feature extract  
1. perdict.py for (two picture compare)  
2. predict_from_dataset.py (input a picture compare the whole dataset,will get a avg similarity)  

### iot_tracker_gui
write in main file

### wait to fix  
Poor ability of siamese network,need light and high accuracy. Deal the problem until i had more time 
