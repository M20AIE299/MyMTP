#Mtech Project 
#Cloud based healthcare monitoring system
#by Saumitra Tarey
#M20AIE299


import streamlit as st
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import psycopg2
from sqlalchemy import create_engine, text
import speech_recognition as sr
import pyttsx3 
import pyautogui
from PIL import Image
import random
import time
#for DL
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
#explainable AI 
import shap

#import warnings
#warnings.filterwarnings("ignore", message="numpy.dtype size changed")
#warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
# Load datasets
#df = pd.read_csv("C:/Testing.csv")
#query the pgsql database to get training data

#global variables
#name = ''
#name2 = ''
#diagnosis = ''
#symptoms = []


# Function to convert text to
# speech
def SpeakText(command):
    
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command) 
    engine.runAndWait()
    
#method to recognize voice command

def recognizespeech(input_type):
    
    # Initialize the recognizer 
    r = sr.Recognizer() 
    
    while(1):    
        
        # Exception handling to handle
        # exceptions at the runtime
        try:
            
            # use the microphone as source for input.
            with sr.Microphone() as source2:
                
                # wait for a second to let the recognizer
                # adjust the energy threshold based on
                # the surrounding noise level 
                r.adjust_for_ambient_noise(source2, duration=0.2)
                
                #listens for the user's input 
                audio2 = r.listen(source2)
                
                # Using google to recognize audio
                MyText = r.recognize_google(audio2)
                if MyText is not None:
                    #MyText = MyText.lower()
                    if input_type == 'Name':
                        name = MyText
                    if input_type == 'Diagnosis':
                        diagnosis = MyText
                    if input_type == 'Name2':
                        name2 = MyText 
                    
                    break
    
                #print('Did you say')
                #SpeakText(MyText)
                #st.success(MyText) 
        except sr.RequestError as e:
            print('Could not request results')
            
        except sr.UnknownValueError:
            print('unknown error occurred')
    return MyText


conn = psycopg2.connect(database = "modeldata", 
                        user = "postgres", 
                        host= 'localhost',
                        password = "12345678",
                        port = 5432)


cur = conn.cursor()
cur.execute('SELECT * FROM "TRAININGDATA";')
rows = cur.fetchall()
print(type(rows))
conn.commit()
conn.close()
#for row in rows:
    #print(type(row))
    
    
# create DataFrame using data
df = pd.DataFrame(rows, columns =['patient_name',
'itching',
'skin_rash',
'nodal_skin_eruptions',
'continuous_sneezing',
'shivering',
'chills',
'joint_pain',
'stomach_pain',
'acidity',
'ulcers_on_tongue',
'muscle_wasting',
'vomiting',
'burning_micturition',
'spotting_urination',
'fatigue',
'weight_gain',
'anxiety',
'cold_hands_and_feets',
'mood_swings',
'weight_loss',
'restlessness',
'lethargy',
'patches_in_throat',
'irregular_sugar_level',
'cough',
'high_fever',
'sunken_eyes',
'breathlessness',
'sweating',
'dehydration',
'indigestion',
'headache',
'yellowish_skin',
'dark_urine',
'nausea',
'loss_of_appetite',
'pain_behind_the_eyes',
'back_pain',
'constipation',
'abdominal_pain',
'diarrhoea',
'mild_fever',
'yellow_urine',
'yellowing_of_eyes',
'acute_liver_failure',
'fluid_overload',
'swelling_of_stomach',
'swelled_lymph_nodes',
'malaise',
'blurred_and_distorted_vision',
'phlegm',
'throat_irritation',
'redness_of_eyes',
'sinus_pressure',
'runny_nose',
'congestion',
'chest_pain',
'weakness_in_limbs',
'fast_heart_rate',
'pain_during_bowel_movements',
'pain_in_anal_region',
'bloody_stool',
'irritation_in_anus',
'neck_pain',
'dizziness',
'cramps',
'bruising',
'obesity',
'swollen_legs',
'swollen_blood_vessels',
'puffy_face_and_eyes',
'enlarged_thyroid',
'brittle_nails',
'swollen_extremeties',
'excessive_hunger',
'extra_marital_contacts',
'drying_and_tingling_lips',
'slurred_speech',
'knee_pain',
'hip_joint_pain',
'muscle_weakness',
'stiff_neck',
'swelling_joints',
'movement_stiffness',
'spinning_movements',
'loss_of_balance',
'unsteadiness',
'weakness_of_one_body_side',
'loss_of_smell',
'bladder_discomfort',
'foul_smell_of_urine',
'continuous_feel_of_urine',
'passage_of_gases',
'internal_itching',
'toxic_look_typhos',
'depression',
'irritability',
'muscle_pain',
'altered_sensorium',
'red_spots_over_body',
'belly_pain',
'abnormal_menstruation',
'dischromic_patches',
'watering_from_eyes',
'increased_appetite',
'polyuria',
'family_history',
'mucoid_sputum',
'rusty_sputum',
'lack_of_concentration',
'visual_disturbances',
'receiving_blood_transfusion',
'receiving_unsterile_injections',
'coma',
'stomach_bleeding',
'distention_of_abdomen',
'history_of_alcohol_consumption',
'fluid_overload_brain',
'blood_in_sputum',
'prominent_veins_on_calf',
'palpitations',
'painful_walking',
'pus_filled_pimples',
'blackheads',
'scurring',
'skin_peeling',
'silver_like_dusting',
'small_dents_in_nails',
'inflammatory_nails',
'blister',
'red_sore_around_nose',
'yellow_crust_ooze',
'prognosis',
'created_date'])

#drop the patient name & created date from the dataset
df = df.drop('patient_name', axis=1)
df = df.drop('created_date', axis=1)

df.replace({'prognosis': {
    'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 
    'Drug Reaction': 4, 'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 
    'Gastroenteritis': 8, 'Bronchial Asthma': 9, 'Hypertension ': 10, 
    'Migraine': 11, 'Cervical spondylosis': 12, 'Paralysis (brain hemorrhage)': 13, 
    'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16, 'Dengue': 17, 'Typhoid': 18, 
    'hepatitis A': 19, 'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 
    'Hepatitis E': 23, 'Alcoholic hepatitis': 24, 'Tuberculosis': 25, 
    'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 
    'Heart attack': 29, 'Varicose veins': 30, 'Hypothyroidism': 31, 
    'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34, 
    'Arthritis': 35, '(vertigo) Paroymsal  Positional Vertigo': 36, 
    'Acne': 37, 'Urinary tract infection': 38, 'Psoriasis': 39, 'Impetigo': 40
}}, inplace=True)

# Prepare features and labels
l1 = l1 = ['itching',
'skin_rash',
'nodal_skin_eruptions',
'continuous_sneezing',
'shivering',
'chills',
'joint_pain',
'stomach_pain',
'acidity',
'ulcers_on_tongue',
'muscle_wasting',
'vomiting',
'burning_micturition',
'spotting_urination',
'fatigue',
'weight_gain',
'anxiety',
'cold_hands_and_feets',
'mood_swings',
'weight_loss',
'restlessness',
'lethargy',
'patches_in_throat',
'irregular_sugar_level',
'cough',
'high_fever',
'sunken_eyes',
'breathlessness',
'sweating',
'dehydration',
'indigestion',
'headache',
'yellowish_skin',
'dark_urine',
'nausea',
'loss_of_appetite',
'pain_behind_the_eyes',
'back_pain',
'constipation',
'abdominal_pain',
'diarrhoea',
'mild_fever',
'yellow_urine',
'yellowing_of_eyes',
'acute_liver_failure',
'fluid_overload',
'swelling_of_stomach',
'swelled_lymph_nodes',
'malaise',
'blurred_and_distorted_vision',
'phlegm',
'throat_irritation',
'redness_of_eyes',
'sinus_pressure',
'runny_nose',
'congestion',
'chest_pain',
'weakness_in_limbs',
'fast_heart_rate',
'pain_during_bowel_movements',
'pain_in_anal_region',
'bloody_stool',
'irritation_in_anus',
'neck_pain',
'dizziness',
'cramps',
'bruising',
'obesity',
'swollen_legs',
'swollen_blood_vessels',
'puffy_face_and_eyes',
'enlarged_thyroid',
'brittle_nails',
'swollen_extremeties',
'excessive_hunger',
'extra_marital_contacts',
'drying_and_tingling_lips',
'slurred_speech',
'knee_pain',
'hip_joint_pain',
'muscle_weakness',
'stiff_neck',
'swelling_joints',
'movement_stiffness',
'spinning_movements',
'loss_of_balance',
'unsteadiness',
'weakness_of_one_body_side',
'loss_of_smell',
'bladder_discomfort',
'foul_smell_of_urine',
'continuous_feel_of_urine',
'passage_of_gases',
'internal_itching',
'toxic_look_typhos',
'depression',
'irritability',
'muscle_pain',
'altered_sensorium',
'red_spots_over_body',
'belly_pain',
'abnormal_menstruation',
'dischromic_patches',
'watering_from_eyes',
'increased_appetite',
'polyuria',
'family_history',
'mucoid_sputum',
'rusty_sputum',
'lack_of_concentration',
'visual_disturbances',
'receiving_blood_transfusion',
'receiving_unsterile_injections',
'coma',
'stomach_bleeding',
'distention_of_abdomen',
'history_of_alcohol_consumption',
'fluid_overload_brain',
'blood_in_sputum',
'prominent_veins_on_calf',
'palpitations',
'painful_walking',
'pus_filled_pimples',
'blackheads',
'scurring',
'skin_peeling',
'silver_like_dusting',
'small_dents_in_nails',
'inflammatory_nails',
'blister',
'red_sore_around_nose',
'yellow_crust_ooze'] # Your symptoms list

disease = ['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']  # Your diseases list

X = df[l1]
y = df[["prognosis"]].values.ravel()

# Streamlit UI
#st.snow()
# adding home tab
tab1,tab2,tab3,tab4 = st.tabs(["Get Prognosis","Add Data","Smartwatch Simulator","Analytics"])
with tab1:
    
    st.title("Disease Predictor using Machine Learning")
    
    # Patient's name
    name = st.text_input("Name of the Patient", key="random1")
    if st.button("talk", key="random8"):
        SpeakText('Please say patient name')
        Speech = recognizespeech('Name')
        print(Speech)
        print('test123')
        st.success(Speech)
    
    # Symptom selection
    symptoms = st.multiselect("Select Symptoms", options=l1, key="random2")
    
    # Decision Tree prediction
    if st.button("Predict Disease", key="random6"):
        l2 = [1 if symptom in symptoms else 0 for symptom in l1]
    
        # Decision Tree Classifier
        clf = tree.DecisionTreeClassifier()
        clf.fit(X, y)
        prediction = clf.predict([l2])
        
        # Show the predicted disease
        predicted_disease = disease[prediction[0]]
        #st.success(f"Predicted Disease: {predicted_disease}")
    
        # Random Forest Classifier
        clf_rf = RandomForestClassifier()
        clf_rf.fit(X, y)
        prediction_rf = clf_rf.predict([l2])
        predicted_disease_rf = disease[prediction_rf[0]]
        #st.success(f"Predicted Disease (Random Forest): {predicted_disease_rf}")
    
        # Naive Bayes Classifier
        gnb = GaussianNB()
        gnb.fit(X, y)
        prediction_nb = gnb.predict([l2])
        predicted_disease_nb = disease[prediction_nb[0]]
        #st.success(f"Predicted Disease (Naive Bayes): {predicted_disease_nb}")
        
        #Get and notify the prediction
        with st.spinner("Getting prediction..."):
            time.sleep(3)
            st.success(f"Predicted Disease: {predicted_disease}")
            st.success(f"Predicted Disease (Random Forest): {predicted_disease_rf}")
            st.success(f"Predicted Disease (Naive Bayes): {predicted_disease_nb}")
        
    if st.button("Reload", key="random18"):
        print('refresh')
        pyautogui.hotkey("ctrl","F5")
        #set all the session variables to null
        

with tab2:
    print('abc')
    st.title("Insert Training Data")
    #function to add single quotes
    def foo2(char):
        return("'{}'".format(char))
    
    #add variable to differentiate the inputs verbal vs text
    if 'is_verbal' not in st.session_state:
        st.session_state.is_verbal = 'No'
    
    # Patient's name
    temp1 = st.text_input("Name of the Patient", key="random3")
    #if 'name2' not in st.session_state:
    name2 = foo2(temp1)
    
    if st.button("talk", key="random9"):
        SpeakText('Please say patient name')
        Speech2 = recognizespeech('Name2')
        print(Speech2)
        print('test1234')
        st.success(Speech2)
        print(Speech2)
        if 'verbal_name2' not in st.session_state:
            st.session_state.verbal_name2 = foo2(Speech2)
            #st.session_state.name2 = None
            print('verbal name2')
            print(st.session_state.verbal_name2)
    
    
    temp2 = st.text_input("Enter confirmed Diagnosis", key="random4")
    #if 'diagnosis' not in st.session_state:
    diagnosis = foo2(temp2)
    
    
    if st.button("talk", key="random10"):
        SpeakText('Please enter the diagnosis')
        Speech3 = recognizespeech('Diagnosis')
        print(Speech3)
        print('test12345')
        st.success(Speech3)
        print(Speech3)
        if 'verbal_diagnosis' not in st.session_state:
            st.session_state.verbal_diagnosis = foo2(Speech3)
            #st.session_state.diagnosis = None
            print('verbal diagnosis')
            print(st.session_state.verbal_diagnosis)
            
        #add variable to differentiate the inputs verbal vs text
        #if 'is_verbal' not in st.session_state:
        st.session_state.is_verbal = 'Yes'
    
    
    
    # Symptom selection
    symptoms2 = st.multiselect("Select Symptoms", options=l1, key="random5")
    print(symptoms2)
    query1 = 'INSERT INTO "TRAININGDATA"('
    temp1 = ''
    for x in symptoms2:
        temp1 = temp1+x+', '
    temp1 = temp1[:-2]    
    query2 = temp1 + ')'
    query3 = ' VALUES('
    temp2 = ''
    for y in symptoms2:
        temp2 = temp2+'1'+', '
    temp2 = temp2[:-2] 
    query4 = temp2 + ');'

    query = query1+query2+query3+query4
    print(query)
    
    if st.button("Sync", key="random7"):
        st.success("Saved Successfully")
        conn2 = psycopg2.connect(database = "modeldata", 
                                user = "postgres", 
                                host= 'localhost',
                                password = "12345678",
                                port = 5432)



        #execute the insert query to insert the training data record 
        cur2 = conn2.cursor()
        cur2.execute(query)                
        conn2.commit()
        #update the patient name and prognosis
        #update_patient_details = 'UPDATE "TRAININGDATA" SET patient_name = ' +'"'+name2+'"'+ ', prognosis = ' +'"'+diagnosis+'"'+ ' ORDER BY created_date DESC LIMIT 1;'
        #print(update_patient_details)
        cur2.execute('SELECT * FROM "TRAININGDATA" ORDER BY created_date DESC LIMIT 1;')
        rows2 = cur2.fetchall()
        date_query = rows2[0][134]
        date_query = foo2(str(date_query))
        print(type(date_query))
        print('look here')
        print(name2)
        print(diagnosis)
        print(type(name2))
        if st.session_state.is_verbal == 'Yes':
            print('if')
            update_name_prognosis = 'UPDATE "TRAININGDATA" SET patient_name = '+st.session_state.verbal_name2+', '+ 'prognosis = '+st.session_state.verbal_diagnosis+' WHERE created_date = '+date_query;
        else:
            print('else')
            update_name_prognosis = 'UPDATE "TRAININGDATA" SET patient_name = '+name2+', '+ 'prognosis = '+diagnosis+' WHERE created_date = '+date_query;
        print(update_name_prognosis)
        cur2.execute(update_name_prognosis)
        conn2.commit()
        conn2.close()
    
    if st.button("Reload", key="random11"):
        print('refresh')
        pyautogui.hotkey("ctrl","F5")
        #set all the session variables to null
        
with tab3:
    #Google Wear OS simulator API 
    #opening the image
    
    col1, col2 = st.columns(2)
    
    path = 'D:/gos.jpg'
    image = Image.open(path)
    #displaying the image on streamlit app
    with(col1):
        st.image(image, caption="Google Wear OS watch")
    
    with(col2):        
        patient_name = st.text_input("Name of the Patient", key="random13")
        patient_name = foo2(patient_name)
        patient_email = st.text_input("Patient email", key="random14")
        patient_email = foo2(patient_email)
        called_doctor = st.text_input("Called Doctor?", key="random15")
        called_doctor = foo2(called_doctor)   
        heart_rate = random.randrange(30, 150)
        body_temp = random.uniform(95.0, 104.0)
        blood_oxygen = random.uniform(90.0, 100.0)
        blood_pressure = random.uniform(90.0, 140.0)
        if st.button("Save", key="random16"):
            print('send')
            #generate data from the smartwatch
              
            print(heart_rate)
            print(body_temp)
            print(blood_oxygen)
            print(blood_pressure)  
            print(patient_name)
            print(patient_email)
            print(called_doctor)
            conn3 = psycopg2.connect(database = "modeldata", 
                                    user = "postgres", 
                                    host= 'localhost',
                                    password = "12345678",
                                    port = 5432)
    
    
            cur3 = conn3.cursor()
            insert_query = 'INSERT INTO smartdata (heart_rate,blood_pressure,blood_oxygen,body_temp,patient_name,patient_email,called_doctor) VALUES ('+str(heart_rate)+','+str(blood_pressure)+','+str(blood_oxygen)+','+str(body_temp)+','+patient_name+','+patient_email+','+called_doctor+');'
            print('insert_query')
            print(insert_query)
            cur3.execute(insert_query)
            conn3.commit()
            conn3.close()
            
        if st.button("Send Data", key="random17"):
            #call the deep learning model to see 
            #if doctor is needed or not
            
            #query training data from postgresdb
            conn4 = psycopg2.connect(database = "modeldata", 
                                    user = "postgres", 
                                    host= 'localhost',
                                    password = "12345678",
                                    port = 5432)


            cur4 = conn4.cursor()
            cur4.execute('SELECT * FROM smartdata ORDER BY created_date;')
            rows4 = cur4.fetchall()
            df2 = pd.DataFrame(rows4, columns =['heart_rate','blood_pressure','blood_oxygen','body_temp','patient_name','patient_email','created_date','called_doctor'])
            df2.drop('patient_name', axis=1)
            df2.drop('patient_email', axis=1)         
            df2.drop('created_date', axis=1)    

    
            # Features (inputs) and target (output)
            X = df2[['heart_rate','blood_pressure','blood_oxygen','body_temp']].values
            z = df2['called_doctor'].replace({'Yes': 1, 'No': 0})
            y = z.values
            #y = df2['called_doctor'].values
    
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
            # Feature scaling (Normalization)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
    
            # Build the Artificial Neural Network (ANN) model
            model = Sequential()
    
            # Input layer and first hidden layer
            model.add(Dense(64, input_dim=4, activation='relu'))  # 4 input features, ReLU activation
    
            # Second hidden layer
            model.add(Dense(32, activation='relu'))  # 32 units, ReLU activation
    
            # Output layer (binary classification, sigmoid activation)
            model.add(Dense(1, activation='sigmoid'))  # Output 1 neuron (binary output)
    
            # Compile the model
            model.compile(loss='binary_crossentropy', 
                          optimizer=Adam(), 
                          metrics=['accuracy'])
    
            # Train the model
            model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))
            


            
    # Evaluate the model on the test set
            loss, accuracy = model.evaluate(X_test, y_test)
            print(f'Test Loss: {loss:.4f}')
            print(f'Test Accuracy: {accuracy:.4f}')
    
            # Make predictions on new data
            #new_data = np.array([[80, 98.7, 97.5, 120]])  # Example input
            new_data = np.array([[heart_rate, body_temp, blood_oxygen, blood_pressure]])
            new_data_scaled = scaler.transform(new_data)
            prediction = model.predict(new_data_scaled)
            
            #xplain the model using shap
            explainer = shap.KernelExplainer(model.predict,X_train)
            shap_values = explainer(new_data_scaled)
            vals = np.abs(shap_values.values).mean(0)
            feature_names = ['heart_rate','blood_pressure','blood_oxygen','body_temp']

            feature_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                 columns=['col_name','feature_importance_vals'])
            feature_importance.sort_values(by=['feature_importance_vals'],
                              ascending=False, inplace=True)
            
            
            cause = feature_importance.values.tolist()
                      
    
            # Output prediction
            print('<<<<output>>>>')
            print(shap_values)
            print(type(shap_values))
            print(feature_importance)
            
            with st.spinner("Getting prediction..."):
                
                time.sleep(3)
                if prediction > 0.5:                
                    st.error('There seems to be an abnormal reading in your '+cause[0][0]+ '. Please call a doctor for consultation')
                    #st.error('Please call a doctor for consultation')
                    
                else:
                    st.success('You are doing fine !')
                
with(tab4):
    
          # Query data to create a trend chart for last 10 readings
    #time = np.linspace(0, 10, 100)  # Time (e.g., in minutes or seconds)
    
    conn5 = psycopg2.connect(database = "modeldata", 
                            user = "postgres", 
                            host= 'localhost',
                            password = "12345678",
                            port = 5432)


    cur5 = conn5.cursor()
    cur5.execute('SELECT heart_rate,blood_pressure,blood_oxygen,body_temp FROM smartdata ORDER BY created_date DESC LIMIT 10;')
    rows5 = cur5.fetchall()
    df3 = pd.DataFrame(rows5, columns =['heart_rate','blood_pressure','blood_oxygen','body_temp'])

    st.line_chart(df3, x_label='Last 10 readings', y_label='health Parameters')
    
    st.bar_chart(df3, x_label='Last 10 readings', y_label='health Parameters')

        
            
            
            
        
        
        
    

    
    

# Note: You may want to handle exceptions and edge cases in a full implementation.
