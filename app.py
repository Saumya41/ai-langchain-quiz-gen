import time
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain import LLMChain, PromptTemplate
import json
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()


# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

groq_api_key= os.getenv('GROQ_API_KEY')

llm=ChatGroq(groq_api_key=groq_api_key,
model_name="Llama3-8b-8192")

# llm = OpenAI(temperature=0.7) ##chatgpt

prompt_template = """
You are a helpful assistant that generates questions. Based on the following topic, generate 5 questions.

Topic: {topic}

Return the questions in a JSON array format.
"""

template = PromptTemplate(input_variables=["topic"], template=prompt_template)
chain = LLMChain(llm=llm, prompt=template)

st.title("AI Question Generator")
st.write("Enter a topic to generate questions.")

def generate_questions(topic):
    # Run the LLM chain with the provided topic
    response = chain.predict(topic=topic)
    # # print("Raw response from the model:\n", response)
    # try:
    #     questions = json.loads(response)
    #     print(questions)
    #     return questions
    # except json.JSONDecodeError:
    #     return {"error": "Failed to generate questions"}
    print(response)
    st.write(response)

def main():
    topic = st.text_input("Input your prompt here")
    if st.button("Generate Questions"):
        with st.spinner("Generating questions..."):
            questions = generate_questions(topic)


if __name__ == "__main__":
    main()

   