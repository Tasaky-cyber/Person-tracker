import numpy as np
import tensorflow as tf
import os 
from PIL import Image

from siamese import Siamese

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
if __name__ == "__main__":
    model = Siamese()
    total_probability = 0
    input_img_path = './input_img.jpg'
    dataset_imgs_path = './dataset'   
    try:
        input_img = Image.open(input_img_path)
    except:
        print('input_img Open Error!!!')
            
    for image in os.listdir(dataset_imgs_path):
        print(image)
        try:
            dataset_img = Image.open(dataset_imgs_path+"/"+image)
        except:
            print('dataset_img Open Error!!!, please check :'+dataset_imgs_path+"/"+image)
        probability = model.detect_image(input_img,dataset_img)
        total_probability = total_probability + probability
        print(probability)
        
    print("avg probability :",total_probability/len(os.listdir(dataset_imgs_path)))
    