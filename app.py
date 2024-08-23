import time
import os
import streamlit as st
# from langchain_groq import ChatGroq
from langchain import LLMChain, PromptTemplate
import json
from langchain_openai.chat_models import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()


openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

model = ChatOpenAI(temperature=0.7, api_key=openai_api_key)


groq_api_key= os.getenv('GROQ_API_KEY')

# llm=ChatGroq(groq_api_key=groq_api_key,
# model_name="Llama3-8b-8192")

# llm = OpenAI(temperature=0.7) ##chatgpt

prompt_template = """
You are a helpful assistant that generates questions. Based on the following topic, generate 5 questions.

Topic: {topic}

Return the questions in a JSON array format.
"""

template = PromptTemplate(input_variables=["topic"], template=prompt_template)
chain = LLMChain(llm=model, prompt=template)

st.title("AI Question Generator")
st.write("Enter a topic to generate questions.")

def generate_questions(topic):
    # Run the LLM chain with the provided topic
    response = chain.predict(topic=topic)
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    if submitted and openai_api_key.startswith("sk-"):
        generate_response(response)


def main():
    topic = st.text_input("Input your prompt here")
    if st.button("Generate Questions"):
        with st.spinner("Generating questions..."):
            questions = generate_questions(topic)


if __name__ == "__main__":
    main()

   