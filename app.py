# Importing Libraries
import streamlit as st
#import numpy as np
#from PIL import Image 
#import tensorflow as tf
# from keras.models import load_model
# import os
# import pandas as pd
#import time
#import cv2

import app1
import app2



PAGES={"Crop Disease Prediction":app1,"About Project":app2}


st.sidebar.title("Choose your option to navigate")
selection=st.sidebar.radio("Go to",list(PAGES.keys()))
page=PAGES[selection]
page.app()
