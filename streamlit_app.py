import streamlit as st 
import altair as alt
# import plotly.express as px 
import requests

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime

# Fxn
def predict_emotions(text, api_url = "http://localhost:8000/predict"):
  results = requests.get(api_url, params={'text': text})
  # print(results)
  results = results.json()
  return results

# def get_prediction_proba(docx):
# 	results = pipe_lr.predict_proba([docx])
# 	return results

# Main Application
def main():
  st.title("Tweet Classifier App")
  menu = ["Home","About"]
  choice = st.sidebar.selectbox("Menu",menu)
  if choice == "Home":
    st.subheader("Home-Depression In Tweets")

    with st.form(key='emotion_clf_form'):
      raw_text = st.text_area("tweet")
      submit_text = st.form_submit_button(label='Submit')

    if submit_text:
      col1, col2 = st.beta_columns(2)

      # Apply Fxn Here
      results = predict_emotions(raw_text)
      prediction = results["prediction"]
      probability = results["confidence"]
      
      # add_prediction_details(raw_text,prediction,np.max(probability),datetime.now())
      with col1:
        st.success("Original Text")
        st.write(raw_text)

        st.success("Prediction")
        st.write(prediction)

        st.success("Probability")
        st.write(probability)

  else:
    st.subheader(f"About, {datetime.now()}")

if __name__ == '__main__':
  main()