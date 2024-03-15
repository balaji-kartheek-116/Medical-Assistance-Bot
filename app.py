import streamlit as st
import numpy as np
from keras.models import load_model
import pickle
import nltk
import random
from PIL import Image
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json


nltk.download('punkt')
nltk.download('wordnet')

# Load your model
model = load_model('medical_chatbot_model.h5')
data_file = open('intents.json').read() # read json file
intents = json.loads(data_file) # load json file

# Load words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    # Tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # Stemming
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # Tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # Bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # Filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result



# Streamlit code

import streamlit as st
from PIL import Image
from gtts import gTTS
import speech_recognition as sr
import time
import os

# Define your predict_class() and getResponse() functions here

# Function to load image
def load_image(image_path):
    try:
        image = Image.open(image_path)
        return image
    except FileNotFoundError:
        st.error("Image not found")
        return None

# Function for text-to-speech conversion
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")

# Function for speech-to-text conversion
def speech_to_text(audio_data):
    recognizer = sr.Recognizer()
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")

# Define correct username and password
CORRECT_USERNAME = "admin"
CORRECT_PASSWORD = "password"  # Change this to your desired password

# Set title and separator
st.title("Medical Assistance ChatBot")
st.markdown("---")

# Authentication
session_state = st.session_state
if "authenticated" not in session_state:
    session_state.authenticated = False

# Check if the user is authenticated
if not session_state.authenticated:
    # Login section
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == CORRECT_USERNAME and password == CORRECT_PASSWORD:
            session_state.authenticated = True
        else:
            st.error("Incorrect username or password")

# If authenticated, proceed with the application
if session_state.authenticated:
    # Logout button
    if st.button("Logout"):
        session_state.authenticated = False

    # Sidebar options
    option = st.sidebar.radio("Choose Interaction Type", ("Text to Text", "Text to Voice"))

    # Load and display the image
    image = load_image("image1.png")
    if image:
        st.image(image)

    # Text-to-Text interaction
    if option == "Text to Text":
        user_input = st.text_input("Enter your message:")
        if st.button("Submit"):
            # Perform chatbot response logic here based on user_input
            ints = predict_class(user_input, model)
            response = getResponse(ints, intents)
            st.success(f"MedBot: {response}")

    elif option == "Text to Voice":
        user_input = st.text_input("Enter your message:")
        if st.button("Submit"):
            # Perform chatbot response logic here based on user_input
            ints = predict_class(user_input, model)
            response = getResponse(ints, intents)
            text_to_speech(response)
            time.sleep(2)
            audio_file_path = "response.mp3"
            if os.path.exists(audio_file_path):
                st.success("Audio file created successfully.")
                st.audio(audio_file_path, format='audio/mp3')
            else:
                st.error("Audio file not found.")