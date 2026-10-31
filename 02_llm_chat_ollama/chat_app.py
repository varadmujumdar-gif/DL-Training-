# import os
# from dotenv import load_dotenv
# The old 'langchain_community.llms' import was removed because it is deprecated.
# We replaced it with 'ChatOllama' from the modern, official 'langchain_ollama' package.
from langchain_ollama import ChatOllama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- STREAMLIT UI SETUP ---
# Kept here as you requested. Streamlit UI code should run early so the interface 
# renders immediately on the screen without waiting for the LLM to load.
st.title("LangChain Demo with gemma2:2b")
input_text = st.text_input("Enter your question here:")


#Creating My Prompts
# --- PROMPT TEMPLATE SETUP ---
# FIXED: Changed 'from_template' to 'from_messages' to correctly parse the role-based list.
# FIXED: Added the missing comma between the system and user tuples to resolve the syntax crash.

prompt = ChatPromptTemplate.from_messages(  # 3. Changed 'from_template' to 'from_messages'
    [
        ("system", "You are a helpful assistant, please respond to the question asked."),
        ("user", "Question:{question}")
    ]
)

#Ollama Model
# --- LLM CONFIGURATION ---
# FIXED: Switched from legacy 'Ollama' to modern 'ChatOllama'.
# FIXED: The modern class uses the parameter 'model=' instead of the old 'model_name='.

llm = ChatOllama(model="gemma2:2b") 
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))