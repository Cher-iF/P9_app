# Import relevant libraries
import streamlit as st
import requests
import json
from PIL import Image

st.header(' ## Article Recommendation system ##')

#azure_url = "https://<your_resource_name>.azurewebsites.net/api/<your_function_name>"
#for example : azure_url = "https://netflix_recommendations.azurewebsites.net/api/get_recommandations/"
azure_url = "https://p9-function-app.azurewebsites.net/api/HttpTrigger1?code=v6N3LQPVgQySrRQYoxYrzxAsP50uFh-WzhqrBriOCAl6AzFu7GwXBw=="

title = st.text_input('put your article id')

if st.button('open your article '):

    if title is not None :
        st.sidebar.header('You can read also ....')
        #st.write('your article below :')
        image = Image.open('C:/Users/AMC/Desktop/p9_stream/articles_example/1.PNG')
        st.image(image, caption='your article')
    
        
        num = int(title)
        # Set the parameters of the request
        request_params = {"user_id":num}

        # Set the parameters of the request
        request_params = {"user_id":num}

        # Send the request to the Azure function
        r = requests.post(azure_url, params=request_params)

        # Grab the recommendations as a Python dictionary
        recommendations_dict = json.loads(r.content.decode())
        
        st.sidebar.write(list(recommendations_dict["recommendations"]))
        