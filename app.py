# Importing Libraries
import streamlit as st
import io
import numpy as np
from PIL import Image 
import tensorflow as tf



import os
import numpy as np
import pandas as pd
import time
import cv2

import app1
import app2


#image_rootpath='Images/'

PAGES={"Crop Disease Prediction":app1,"About Project":app2}


# In[49]:


st.sidebar.title("Choose your option to navigate")
selection=st.sidebar.radio("Go to",list(PAGES.keys()))
page=PAGES[selection]
page.app()