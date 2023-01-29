import pickle
from pathlib import Path

import streamlit_authenticator 

names = ['Hrithik Maddirala', 'Gadaputi Ashritha']
usernames = ['hrithikm2002' , 'gashritha']
passwords = ['abc123', 'def456']

hashed_passwords = streamlit_authenticator.Hasher(passwords).generate()

fp = Path("C:\\Users\\Anil\\Downloads\\User_Interface\\hashed_passwords.pkl")

with fp.open("wb") as file : 
	pickle.dump(hashed_passwords, file)