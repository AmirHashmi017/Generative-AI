from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
model= ChatOpenAI(model='gpt-4')

# Web app UI
st.header("Research Tool")

user_input= st.text_input("Enter Your Prompt")

if st.button("Summarize"):
    result= model.invoke(user_input)
    st.write(result.content)

