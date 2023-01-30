import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
#import keras 
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.utils import load_img,img_to_array,array_to_img
#from joblib import load
#import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
#from keras import backend as K
#import pywhatkit
#from datetime import datetime
from PIL import Image
import pickle
from pathlib import Path
import streamlit_authenticator
import mysql.connector
import random

streamlit_style = """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap');
            html, body, [class*="css"]  {
            font-family: 'Georgia', sans-serif;
            }
            </style>
            """

st.markdown(streamlit_style, unsafe_allow_html=True)

#Database

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="Ashri@2003",
    database="pets"
)

#Authentication 

names = ['Hrithik Maddirala', 'Gadaputi Ashritha']
usernames = ['hrithikm2002' , 'gashritha']

fp = Path("hashed_passwords.pkl")

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
    menu = ["🏠 Home","🍞 Food","🙍 Information"]

    authenticator.logout("Logout", "sidebar")
    st.sidebar.title(f"Welcome {name}")
    choice = st.sidebar.selectbox("MENU",menu)
    if choice == "🏠 Home":
        st.subheader("THE NEW AGE FOOD RECOMMENDATION")
        st.markdown("Rather than going through the trouble of recommending customers what to order, what about using an emotion-based food recommendation app which will recommend customers food based on how they are feeling? Welcome to the one-stop website for all your food needs!")
        img = Image.open('home_pic.jpg') 
        st.image(img,width=600)

    if choice == "🍞 Food":
        st.subheader("HELLO!")
        menu1 = ["Emotion-based Food Recommendation"] 
        ch = st.selectbox("Select an option",menu1)
        st.text("")
        if ch == "Emotion-based Food Recommendation":
            faceDetect=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            model = tf.keras.models.load_model("final_model_file.h5")
            ### load file
            uploaded_file = st.file_uploader("Choose an image file", type="jpg")
        
            map_dict = {0: 'Negative',
                    1: 'Neutral',
                    2: 'Positive'}
            
            cursor = conn.cursor()
            cursor.execute(" insert into result_table(foodname,flavor) select name,flavor from food where not exists(select * from employee where employee.a1=food.ing1  union select * from employee where employee.a1=food.ing2 union select * from employee where employee.a1=food.ing3 union select * from employee where employee.a1=food.ing4);")
            cursor.execute("select * from result_table")
            results = cursor.fetchall()
            
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
                    x,y,w,h = faces[0]
                    # for x,y,w,h in faces[0][0]:
                    sub_face_img=gray[y:y+h, x:x+w]
                    resized=cv2.resize(sub_face_img,(48,48))
                    normalize=resized/255.0
                    reshaped=np.reshape(normalize, (1, 48, 48, 1))
                    result=model.predict(reshaped)
                    label=np.argmax(result, axis=1)[0]
                    # prediction = model.predict(img_reshape).argmax()
                    st.subheader("Predicted label for the image is {}".format(map_dict [label]))    
                    st.text("")
                    if label==0:
                       cursor.execute("select distinct foodname from result_table where flavor='Protein/Fiber'") 
                       results1=cursor.fetchall()
                       size = len(results1)
                       var = random.randint(0,size-1)
                       st.subheader(str(results1[var][0]))
                    if label==1:
                       cursor.execute("select distinct foodname from result_table where flavor='Cereals'") 
                       results1=cursor.fetchall()
                       size = len(results1)
                       var = random.randint(0,size-1)  
                       st.subheader(str(results1[var][0]))
                    if label==2:
                       cursor.execute("select distinct foodname from result_table where flavor='Fast foods' or flavor='sweets'") 
                       results1=cursor.fetchall()
                       size = len(results1)
                       var = random.randint(0,size-1)
                       st.subheader(str(results1[var][0]))
                    #cursor = conn.cursor()
                    #cursor.execute(" insert into result_table(foodname,flavor) select name,flavor from food where not exists(select * from employee where employee.a1=food.ing1  union select * from employee where employee.a1=food.ing2 union select * from employee where employee.a1=food.ing3 union select * from employee where employee.a1=food.ing4);")
                    #cursor.execute("select * from result_table")
                    #results = cursor.fetchall()
                       #st.markdown(results1)

    if choice=="🙍 Information":

        menu2 = ["Informative Plots", "FAQ's"] 
        ch = st.selectbox("Select an option",menu2)
        if ch=="Informative Plots":
            menu3 = ["Calories","Protein","Fat","Saturated Fat","Fiber","Carbohydrates"] 
            ch2 = st.selectbox("Select an option",menu3)
            st.text("")
            df=pd.read_csv("final_dataset.csv")
            
            if ch2 == "Calories":
                df1 = df.sort_values('Actual_Calories',ascending = False)[0:20]
                fig1=px.bar(df1,x='Food',y='Actual_Calories')
                st.plotly_chart(fig1)
            if ch2 == "Protein":
                df2 = df.sort_values('Actual_Protein',ascending = False)[3:22]
                fig2=px.bar(df2,x='Food',y='Actual_Protein')
                st.plotly_chart(fig2)
            if ch2 == "Fat":
                df3 = df.sort_values('Actual_Fat',ascending = False)[9:20]
                fig3=px.bar(df3,x='Food',y='Actual_Fat')
                st.plotly_chart(fig3)
            if ch2 == "Saturated Fat":
                df4 = df.sort_values('Actual_Sat.Fat',ascending = False)[3:14]
                fig4=px.bar(df4,x='Food',y='Actual_Sat.Fat')
                st.plotly_chart(fig4)
            if ch2 == "Fiber":
                df5 = df.sort_values('Actual_Fiber',ascending = False)[3:14]
                fig5=px.bar(df5,x='Food',y='Actual_Fiber')
                st.plotly_chart(fig5)
            if ch2 == "Carbohydrates":
                df6 = df.sort_values('Actual_Carbs',ascending = False)[5:16]
                fig6=px.bar(df6,x='Food',y='Actual_Carbs')
                st.plotly_chart(fig6)

        if ch=="FAQ's":

            if st.button('Is there any connection between food and mood?'):
                st.markdown('There have been many studies conducted which state that the food we eat influences our mental health.')
            
            if st.button('How do we use the recommendation system?'):
                st.markdown("Select the Food option on the sidebar. Upload a photo of the customer, and based on the customer's mood & other details like customer allergies,etc ,few food items will be recommended for them.")
            image_faq = Image.open("FAQ.jpg") 
            st.image(image_faq,width=650)  
            
conn.close()    

        
