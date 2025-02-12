import os 
from constants import openai_api
from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
import streamlit as st



st.title('Celebrity Search Results')
input_text=st.text_input("Search the Celebrity you want to know about !!!!!")


#Functions
def generate_celebrity_info(name: str) -> str:
    return f"Details about {name}"

def generate_birth_date(person_info: str) -> str:
    return f"{person_info}\nBirth date:..."

def generate_world_events(inputs: dict) -> str:
    dob = inputs.get("dob", "unknown date")  
    return f"5 major global events around {dob}:"

#LLM
llm = OpenAI(temperature=0.8)

#prompt templates
first_prompt = PromptTemplate(
    input_variables=["name"],
    template="Tell me about celebrity {name}"
)
first_input_runnable = RunnableLambda(generate_celebrity_info)
chain = first_input_runnable | llm
output = chain.invoke({"input": "person"})

second_input_prompt=PromptTemplate(
    input_variables=["person"],
    template="When was {person} born"
)
second_input_runnable = RunnableLambda(generate_birth_date)
chain2 = second_input_runnable | llm
output = chain.invoke({"input": "dob"})

third_input_prompt=PromptTemplate(
    input_variables=["dob"],
    template="Mention 5 major events happend aroud {dob} in the world"
)
third_input_runnable = RunnableLambda(generate_world_events)
chain3 = third_input_runnable | llm
output = chain.invoke({"input": "dob"})

parent_chain = chain | chain2 | chain3
output = parent_chain.invoke(input_text)
print(output)

if input_text:
    st.write(chain.run(input_text))