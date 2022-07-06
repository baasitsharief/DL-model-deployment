import streamlit as st 
import altair as alt
import requests

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime
import plotly.graph_objects as go

# Fxn
def predict_emotions(text, api_url = "http://localhost:8000/predict"):
  results = requests.get(api_url, params={'text': text})
  results = results.json()
  return results

def get_explanations(text, api_url = "http://localhost:8000/explanation"):
  exp_json = requests.get(api_url, params={'text': text})
  exp_as_list = exp_json.json()
  words = list()
  weights = list()
  for pair in exp_as_list:
    words.append(pair[0])
    weights.append(pair[1])
  return words, weights

# Main Application
def main():
  st.title("Tweet Classifier App")
  menu = ["Home","About"]
  choice = st.sidebar.selectbox("Menu",menu)
  if choice == "Home":
    st.subheader("Home\nDepression In Tweets")

    with st.form(key='depression_clf_form'):
      raw_text = st.text_area("tweet")
      submit_text = st.form_submit_button(label='Submit')
      # exp_button = st.form_submit_button(label = "Submit and Get Explanations")

    if submit_text:
      col1, col2 = st.columns(2)

      results = predict_emotions(raw_text)
      prediction = results["prediction"]
      probability = results["confidence"]

      with col1:
        st.success("Original Text")
        st.write(raw_text)

        st.success("Prediction")
        st.write(prediction)

        st.success("Probability")
        st.write(probability)

      # with col2:
      #   if exp_button:
      #     words, weights = get_explanations(raw_text)
      #     st.success("Explanation")
      #     fig = go.Figure(go.Bar(
      #       x=weights,
      #       y=words,
      #       orientation='h'))
      #     st.plotly_chart(fig, use_container_width=True)

  else:
    st.subheader(f"About, {datetime.now()}")

if __name__ == '__main__':
  main()