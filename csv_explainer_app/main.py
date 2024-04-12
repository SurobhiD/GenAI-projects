from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
#from pandasai.llm import BambooLLM

load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']
llm = OpenAI(api_token=API_KEY)

#API_KEY = os.environ['BAMBOO_API_KEY']
#llm = BambooLLM(api_key=API_KEY)

st.title("Prompt-driven CSV analysis")
uploaded_file = st.file_uploader("Upload a csv file", type = ['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head(3))

    prompt = st.text_area("Enter your prompt:")

    if st.button("Generate the answer"):
        if prompt:
            with st.spinner("Generating response..."):
                sdf = SmartDataframe(df, config={"llm": llm})
                response = sdf.chat(prompt)
                st.write(response)
        else:
            st.warning("Please enter a prompt")
