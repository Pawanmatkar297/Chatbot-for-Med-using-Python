import speech_recognition as sr
import pyttsx3
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import re
import pickle
import os
import tensorflow as tf

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class HealthcareChatbot:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.intent_classifier = RandomForestClassifier(n_estimators=100)
        self.response_generator = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        self.intents_df = self.load_intents('Diseases_Symptoms.csv')
        self.load_or_train_models()

    def load_intents(self, file_path):
        df = pd.read_csv(file_path)
        print("Columns in the CSV file:", df.columns)
        print("First few rows of the dataframe:")
        print(df.head())
        return df

    def load_or_train_models(self):
        if os.path.exists('intent_classifier.pkl') and os.path.exists('vectorizer.pkl'):
            self.intent_classifier = pickle.load(open('intent_classifier.pkl', 'rb'))
            self.vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
        else:
            self.train_intents()

    def train_intents(self):
        if 'Symptoms' not in self.intents_df.columns or 'Name' not in self.intents_df.columns:
            raise ValueError(f"Required columns 'Symptoms' and 'Name' not found in the CSV. Columns present: {self.intents_df.columns}")
        
        X = self.intents_df['Symptoms'].tolist()
        y = self.intents_df['Name'].tolist()
        
        X_vectorized = self.vectorizer.fit_transform(X)
        self.intent_classifier.fit(X_vectorized, y)
        
        # Save models
        pickle.dump(self.intent_classifier, open('intent_classifier.pkl', 'wb'))
        pickle.dump(self.vectorizer, open('vectorizer.pkl', 'wb'))

    def speech_to_text(self):
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
            try:
                text = self.recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand that.")
                return None
            except sr.RequestError:
                print("Sorry, there was an error with the speech recognition service.")
                return None

    def text_to_speech(self, text):
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def preprocess_text(self, text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def classify_intent(self, text):
        preprocessed_text = self.preprocess_text(text)
        vectorized_text = self.vectorizer.transform([preprocessed_text])
        intent = self.intent_classifier.predict(vectorized_text)[0]
        return intent

    def generate_response(self, intent_name):
        condition = self.intents_df[self.intents_df['Name'] == intent_name].iloc[0]
        response = f"Based on your symptoms, it could be {condition['Name']}. Common symptoms include {condition['Symptoms']}. Typical treatments involve {condition['Treatments']}. Please consult a healthcare professional for proper diagnosis and treatment."
        return response

    def chat(self):
        print("Healthcare Chatbot: Hello! How can I assist you today? Please describe your symptoms. (Say 'quit' to exit)")
        while True:
            user_input = self.speech_to_text()
            if user_input:
                if user_input.lower() == 'quit':
                    print("Healthcare Chatbot: Goodbye! Take care.")
                    self.text_to_speech("Goodbye! Take care.")
                    break
                
                intent = self.classify_intent(user_input)
                response = self.generate_response(intent)
                print(f"Healthcare Chatbot: {response}")
                self.text_to_speech(response)

if __name__ == "__main__":
    try:
        chatbot = HealthcareChatbot()
        chatbot.chat()
    except Exception as e:
        print(f"An error occurred: {str(e)}")