import numpy as np
import pickle
import pandas as pd
# from flasgger import Swagger
import streamlit as st


# app=Flask(__name__)
# Swagger(app)

pickle_in = open("Maj_proj_model_pickle", "rb")
classifier = pickle.load(pickle_in)


# @app.route('/')
#def welcome():
#    return "Welcome All"

# @app.route('/predict',methods=["Get"])
def Diabetes_prediction(Preg,Gluc,BP,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    prediction = classifier.predict([[Preg,Gluc,BP,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    print(prediction)
    return prediction


def main():
    st.title("Diabetes Predictor")
    html_temp = """
    <div style="background-color:#546beb;padding:10px">
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    Preg = st.text_input("Pregnancies","")
    Gluc = st.text_input("Glucose","")
    BP = st.text_input("BloodPressure","")
    SkinThickness = st.text_input("SkinThickness","")
    Insulin = st.text_input("Insulin","")
    BMI = st.text_input("BMI","")
    DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction","")
    Age = st.text_input("Age","")
    result = ""
    if st.button("Predict"):
        result = Diabetes_prediction(int(Preg),int(Gluc),int(BP),int(SkinThickness),int(Insulin),float(BMI), float(DiabetesPedigreeFunction),int(Age))
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")


#if __name__ == '__main__':
main()
