import os
from constants import openai_key
from langchain.llms import OpenAI

import streamlit as st

#OpenAI API
os.environ["OPENAI_API_KEY"]=openai_key

#streamlitFramework

st.title('Langchain Demo With OPENAI API')
input_text=st.text_input("Search the topic u want")

#LLMS
llm=OpenAI(temperature=0.8)


if input_text:
    st.write(llm(input_text))