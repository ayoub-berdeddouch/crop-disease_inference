import streamlit as st
# import matplotlib.pyplot as plt
from PIL import Image
import numpy as np





def app():
    st.title("About us")
    img=Image.open("Images/omdena.png")
    newsize=(280,300)
    img1=img.resize(newsize)
    st.subheader("1.Omdena Kenya Chapter")
    st.subheader("Collaborate, learn, build, grow.")
    st.image(img1)
    st.write("\n")
    st.write("\n")
    st.write("\n")
    img2=Image.open("Images/omdena_kenya.jpg")
    img2=img2.resize(newsize)
    st.subheader(" Crop Disease Prediction ππππ")
    st.image(img2)
    st.write("\n")
    st.write("\n")
    st.markdown(" ### Model used : Efficient-Net")
    st.markdown(" ### Model Accuracy : 99.65%")
    st.markdown(" ### 38 were used for Training the Model.")
    st.markdown(" #  Trained and Deployed by: ")
    st.markdown(" ###  π¨π»Arudhra Narasimhan π¨π»βπ» ")
    st.markdown(" ###  π¨π»Ayoub Berdeddouch π¨π»βπ»")
    img3=Image.open("Images/train_validation.PNG")
    #img3=img3.resize(newsize)
    st.subheader("Train Validaation")
    st.image(img3)
    st.write("\n")
    img4=Image.open("Images/cm_matrix.PNG")
    #img4=img4.resize(newsize)
    st.subheader("Classification Report")
    st.image(img4)
    st.write("\n")
    img5=Image.open("Images/errors_test.PNG")
    #img5=img5.resize(newsize)
    st.subheader("Test Errors")
    st.image(img5)

    
