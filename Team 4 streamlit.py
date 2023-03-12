


import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image




# loading in the model to predict on the data




pickle_in = open('RF_model.pkl', 'rb')
RF_model = pickle.load(pickle_in)
RF_model




def welcome():
    return 'welcome all'




# defining the function which will make the prediction using 
# the data which the user inputs




def prediction(industrial_risk, management_risk, financial_flexibility, credibility,competitiveness,operating_risk):  
   
    prediction = RF_model.predict(
        [[industrial_risk, management_risk, financial_flexibility, credibility,competitiveness,operating_risk]])
    print(prediction)
    return prediction




# this is the main function in which we define our webpage 
def main():
    # giving the webpage a title
    st.title("Bankruptcy-Prevention")
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Bankruptcy-Prevention Classifier ML App </h1>
    </div>
    """
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
      
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction
    industrial_risk = st.text_input("industrial_risk", " ")
    management_risk = st.text_input("management_risk", " ")
    financial_flexibility = st.text_input("financial_flexibility", " ")
    credibility = st.text_input("credibility", " ")
    competitiveness = st.text_input("competitiveness", " ")
    operating_risk = st.text_input("operating_risk", " ")
       
    result =""
      
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    if st.button("Predict"):
        result = prediction(industrial_risk, management_risk, financial_flexibility, credibility,competitiveness,operating_risk)
    st.success('The output is {}'.format(result))
     
if __name__=='__main__':
    main()








