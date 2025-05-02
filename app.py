import streamlit as st
import pickle
import pandas as pd
import requests


def charger_modele():
    url = "https://www.dropbox.com/scl/fi/nzk2ejr8a5c7ws2qb8d8r/random_forest_model.pkl?rlkey=2sob7vo4i0l37vmu9xp51snh2&st=1ghra874&dl=0"
    response = requests.get(url)
    rfg = pickle.loads(response.content)
    return rfg

# def charger_transformation():
#     with open('scaler.pkl', 'rb') as fichier_transformation:
#         scaler = pickle.load(fichier_transformation)
#     return scaler

def charger_transformation():
    url = "https://www.dropbox.com/scl/fi/91aro32hn0s6i0mjuv1tn/scaler.pkl?rlkey=uezpvhlk6rkynl4flhg08tei2&st=3un7p9y9&dl=0"
    response = requests.get(url)
    scaler = pickle.loads(response.content)
    return scaler


def predict(bedrooms, bathrooms, sqft_living, view, grade, sqft_basement, sqft_living15):

    input_data = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'sqft_living': [sqft_living],
        'view': [view],
        'grade': [grade],
        'sqft_basement': [sqft_basement],
        'sqft_living15': [sqft_living15]
    })
    
    print(input_data)

    scaler = charger_transformation()
    data_scaled = scaler.transform(input_data)
    data_scaled = pd.DataFrame(data_scaled, columns= input_data.columns)
    print(data_scaled)

    rfg = charger_modele()
    
    prediction = rfg.predict(data_scaled)
    
    st.markdown(
        f"<p style='font-size:24px; font-weight:bold;'>Le prix de la maison est : {prediction[0]}</p>", 
        unsafe_allow_html=True
    )




def main():
    
    st.title("Application de prédiction des prix des maisons")
        
    bedrooms = st.number_input('Nombre de chambres', min_value=0, max_value=6, value=0)
    bathrooms = st.number_input('Nombre de salles de bains', min_value=0.0, max_value=5.0, value=0.0, step=0.5,  format="%.2f")
    sqft_living = st.number_input('Surface habitable', min_value=0.0, max_value=15000.0, value=0.0 )
    view = st.slider('Vue',  min_value=0, max_value=4, value=0)
    grade = st.slider('Qualité',  min_value=1, max_value=13, value=6)
    sqft_basement =  st.number_input('Surface sous-sol', min_value=0.0, max_value=5000.0, value=0.0 )
    sqft_living15 = st.number_input('Surface habitable moyenne voisinage', min_value=0.0, max_value=10000.0, value=0.0 )
    
    if st.button("Predict"):
        predict(bedrooms, bathrooms, sqft_living, view, grade, sqft_basement, sqft_living15)


if __name__ == "__main__":
    main()
