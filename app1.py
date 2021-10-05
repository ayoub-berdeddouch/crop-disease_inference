# Importing Libraries
import streamlit as st
import io
import numpy as np
from PIL import Image 
#import tensorflow as tf
from keras.models import load_model



import os
import numpy as np
import pandas as pd
import time
import cv2



def app():
    # Title and Description

    st.title('Omdena - Kenya Chapter')
    st.markdown('## Crop Disease Prediction 🍁🍃🍂🍀')
    #st.subheader("Model used : Efficient-Net -- Accuracy : 99.65% ")
    #st.subheader("Model ")
    #st.markdown(' ### 👨🏻‍💻 By Arudhra Narasimhan & 👨🏻‍💻Ayoub Berdeddouch')
    st.write("Just Upload your Plant's Leaf Image and Get Predictions if Your Plant is Healthy or Not!")
    st.write("")



    # Loading Model
    #model = tf.keras.models.load_model("model.h5")
    model = load_model("model.h5")

    # Upload the image
    uploaded_file = st.file_uploader("Choose a Image file", type=['jpg','jpeg','png','webp', 'jfif'])

    #st.write("Model is being loaded- this will take about 10 seconds")

    df = pd.read_csv("class_dict.csv")

    #predictions_map = {0:"is healthy", 1:"has Multiple Diseases", 2:"has Rust", 3:"has Scab"}

    class_names = [
                df.class_name[0],
                df.class_name[1],
                df.class_name[2],
                df.class_name[3],
                df.class_name[4],
                df.class_name[5],
                df.class_name[6],
                df.class_name[7],
                df.class_name[8],
                df.class_name[9],
                df.class_name[10],
                df.class_name[11],
                df.class_name[12],
                df.class_name[13],
                df.class_name[14],
                df.class_name[15],
                df.class_name[16],
                df.class_name[17],
                df.class_name[18],
                df.class_name[19],
                df.class_name[20],
                df.class_name[21],
                df.class_name[22],
                df.class_name[23],
                df.class_name[24],
                df.class_name[25],
                df.class_name[26],
                df.class_name[27],
                df.class_name[28],
                df.class_name[29],
                df.class_name[30],
                df.class_name[31],
                df.class_name[32],
                df.class_name[33],
                df.class_name[34],
                df.class_name[35],
                df.class_name[36],
                df.class_name[37],
                ]

    # healthy classes.
    healthy_cls = [3,4,10,14,17,19,22,23,24,27,37]



    if st.button("Process"):
            test_image = Image.open(uploaded_file)
            test_image = np.array(test_image.convert('RGB'))
            test_image_r = cv2.resize(test_image, (224,224),interpolation=cv2.INTER_NEAREST)
            st.image(test_image,caption="Your input image")
            test_image_r = np.expand_dims(test_image_r,axis=0)
            st.write("Processing the image for prediction...")
            
            progress = st.progress(0)
            progress_text = st.empty()
            
            for i in range(101):
                time.sleep(0.2)
                progress.progress(i)
                progress_text.text(f"Progress:{i}%")
                
            probs = model.predict(test_image_r)
            pred_class = np.argmax(probs)

            pred_class_name = class_names[pred_class]
            
            if pred_class in healthy_cls:
                msg = f'Your crop-leaf is healthy and predicted class is {pred_class_name} with probability of {int(probs[0][pred_class]*100)}%'
                st.success(msg)
            else:
                msg = f'Your crop-leaf is not healthy and predicted class is {pred_class_name} with probability of {int(probs[0][pred_class]*100)}%'
                st.error(msg)

            
        
        
        # # result_text = f"The plant leaf {predictions_map[predictions]} with {int(predictions_arr[0][predictions]*100)}% probability"
        
        # ####
        # result_text = f"The plant leaf belong to class: {class_map[predictions]} with a probability of {int(predictions_arr[0][predictions]*100)}%"
        # #####
        # test = {3,4,10,14,17,19,22,23,24,27,37}
        # if predictions in test:
        #     st.success(result_text)
        # else:
        #     st.error(result_text)

