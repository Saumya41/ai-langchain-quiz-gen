import os
import json
from dotenv import load_dotenv
from langchain import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Retrieve the Groq API key from environment variables
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize the LLM with Groq API key and model name
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define a prompt template for generating questions
prompt_template = """
You are a helpful assistant that generates questions. Based on the following topic, generate 5 questions.

Topic: {topic}

Return the questions in a JSON array format.
"""

# Create the prompt template
template = PromptTemplate(input_variables=["topic"], template=prompt_template)

# Create the LLMChain
chain = LLMChain(llm=llm, prompt=template)

def extract_json_from_response(response):
    # Find the start of the JSON array in the response
    start_index = response.find("[")
    if start_index != -1:
        # Extract the JSON array
        json_part = response[start_index:]
        return json_part
    return None

def generate_questions(topic):
    # Invoke the LLM chain with the provided topic
    response = chain.predict(topic=topic)
    
    # Log the raw response for debugging
    print("Raw response from the model:\n", response)

    # Extract the JSON part of the response
    json_part = extract_json_from_response(response)
    
    if json_part:
        # Attempt to parse the extracted JSON
        try:
            questions = json.loads(json_part)
            print(questions)
            return questions
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON."}
    else:
        return {"error": "Failed to extract JSON from response."}

if __name__ == "__main__":
    topic = "Artificial Intelligence in Education"
    questions = generate_questions(topic)
    print(json.dumps(response, indent=2))
