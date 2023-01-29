import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import keras 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img,img_to_array,array_to_img
from joblib import load
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from keras import backend as K
import pywhatkit
from datetime import datetime
from PIL import Image
import pickle
from pathlib import Path
import streamlit_authenticator

#Authentication 

names = ['Hrithik Maddirala', 'Gadaputi Ashritha']
usernames = ['hrithikm2002' , 'gashritha']

fp = Path("C:\\Users\\Anil\\Downloads\\User_Interface\\hashed_passwords.pkl")

with fp.open("rb") as file : 
    hashed_passwords = pickle.load(file)

credentials = {
        "usernames":{
            usernames[0]:{
                "name":names[0],
                "password":hashed_passwords[0]
                         },
            usernames[1]:{
                "name":names[1],
                "password":hashed_passwords[1]
                         }            
                    }       
                }   

authenticator = streamlit_authenticator.Authenticate(credentials,"cookie", "abcdef", cookie_expiry_days= 1)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status : 
    st.title("🍔 EMOTION-BASED FOOD RECOMMENDATION SYSTEM")
    menu = ["🏠 Home","🍞 Food","🙍 Patient"]

    authenticator.logout("Logout", "sidebar")
    st.sidebar.title(f"Welcome {name}")
    choice = st.sidebar.selectbox("MENU",menu)
    if choice == "🏠 Home":
        st.subheader("THE NEW AGE FOOD RECOMMENDATION")
        st.markdown("Rather than going through the trouble of recommending customers what to order, what about using an emotion-based food recommendation app which will recommend customers food based on how they are feeling? Welcome to the one-stop website for all your food needs!")
        image = Image.open('C:\\Users\\Anil\\Downloads\\User_Interface\\home_pic.jpg') 
        st.image(image,width=600)

    if choice == "🍞 Food":
        st.subheader("HELLO!")
        menu1 = ["Emotion-based Food Recommendation"] #"Informative Plots"
        ch = st.selectbox("Select an option",menu1)
        st.text("")
        if ch == "Emotion-based Food Recommendation":
            faceDetect=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            model = tf.keras.models.load_model("C:\\Users\\Anil\\Downloads\\User_Interface\\final_model_file.h5")
            ### load file
            uploaded_file = st.file_uploader("Choose an image file", type="jpg")
        
            map_dict = {0: 'Negative',
                    1: 'Neutral',
                    2: 'Positive'}

            if uploaded_file is not None:
            # Convert the file to an opencv image.
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)
                opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
                # resized = cv2.resize(opencv_image,(224,224))
                # Now do something with the image! For example, let's display it:
                st.image(opencv_image, channels="RGB")

                gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

                Generate_pred = st.button("Generate Prediction")    
                if Generate_pred:
                    faces= faceDetect.detectMultiScale(gray, 1.3, 3)
                    for x,y,w,h in faces:
                        sub_face_img=gray[y:y+h, x:x+w]
                        resized=cv2.resize(sub_face_img,(48,48))
                        normalize=resized/255.0
                        reshaped=np.reshape(normalize, (1, 48, 48, 1))
                        result=model.predict(reshaped)
                        label=np.argmax(result, axis=1)[0]
                        # prediction = model.predict(img_reshape).argmax()
                        st.title("Predicted label for the image is {}".format(map_dict [label]))
                        st.text("")

