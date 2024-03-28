import os
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array


current_path = os.getcwd()
predictor_model = tf.keras.models.load_model(current_path+'/Model/TransferLearning/inceptionv2_1.h5')



print("Requirements loaded")

def predictor(img_path):
    img = load_img(img_path, target_size=(224,224,3))
    print(img_path)
    img = img_to_array(img)
    img = np.expand_dims(img,axis = 0)
    prediction = predictor_model.predict(img)
    prediction = (prediction > 0.5).astype(int)
    return(prediction)
