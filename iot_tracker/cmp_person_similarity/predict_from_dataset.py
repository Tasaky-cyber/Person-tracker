import numpy as np
import tensorflow as tf
import os 
from PIL import Image
from tensorflow import keras
from .nets.siamese import siamese

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
#run stand by
input_shape = (60, 120, 3)
model = siamese(input_shape)
model_path = os.path.expanduser('./cmp_person_similarity/logs/ep070-loss0.079-val_loss0.120.h5')
assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
model.load_weights(model_path)
#force to initial first
model.predict([np.random.rand(1, 60, 120, 3), np.random.rand(1, 60, 120, 3)])

def similarity_detect(search_img):

    total_probability = 0
    input_img_path = './cmp_person_similarity/input_img.jpg'#'./input_img.jpg'
    dataset_imgs_path = './temp'
    #C:\Users\Alex\PycharmProjects\iot_tracker\cmp_person_similarity\dataset
    try:
        #input_img = Image.open(input_img_path)
        #input_img= input_img.resize((105,105))
        input_img = search_img.resize((120, 60))#h,w
        #print("input_imgsize: ", input_img.size)
    except:
        print('input_img Open Error!!!')

    for image in os.listdir(dataset_imgs_path):
        #print(image)
        try:
            dataset_img = Image.open(dataset_imgs_path + "/" + image)
            dataset_img = dataset_img.resize((120,60))
            #print("dataset_imgsize: ",dataset_img.size)
        except:
            print('dataset_img Open Error!!!, please check : ' + dataset_imgs_path + "/" + image)
        #probability = model.detect_image(input_img, dataset_img)

        image_1 = np.asarray(input_img).astype(np.float64) / 255
        image_2 = np.asarray(dataset_img).astype(np.float64) / 255

        if input_shape[-1] == 1:
            image_1 = np.expand_dims(image_1, -1)
            image_2 = np.expand_dims(image_2, -1)

        photo1 = np.expand_dims(image_1,0)
        photo2 = np.expand_dims(image_2,0)
        probability = model.predict([photo1,photo2])
        total_probability = total_probability + probability
        #print(probability)
    avg_probability = total_probability / len(os.listdir(dataset_imgs_path))
    #print("avg probability :", avg_probability)
    return avg_probability#probability

'''    
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
'''