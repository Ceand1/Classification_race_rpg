import pandas as pd
import numpy as np 
import itertools
import keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import Sequential 
from keras import optimizers
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense  
from keras import applications  
from keras.utils.np_utils import to_categorical  
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import math  
import datetime
import time
from keras.models import load_model

vgg16 = applications.VGG16(include_top=False, weights='imagenet') 
model = load_model('bottleneck_fc_model.h5')

def read_image(file_path):
    print("[INFO] loading and preprocessing image...")  
    image = load_img(file_path, target_size=(224, 224))  
    image = img_to_array(image)  
    image = np.expand_dims(image, axis=0)
    image /= 255.  
    return image


def test_single_image(path):
    races = ['cyclops', 'dragonborn', 'elf', 'human', 'orc', 'tieflings']
    images = read_image(path)
    time.sleep(.5)
    bt_prediction = vgg16.predict(images)  
    preds = model.predict(bt_prediction)
    for idx, race, x in zip(range(0,6), races , preds[0]):
        print("ID: {}, Label: {} {}%".format(idx, race, round(x*100,2) ))


path = 'database/test/tieflings/tumblr_bac1a7cf029c0a449c2a500034e908f8_a7a74da3_500.jpg'

test_single_image(path)
