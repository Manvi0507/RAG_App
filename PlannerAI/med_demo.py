# Import all the libraries
from dotenv import load_dotenv
import os
import google.generativeai as genai
import streamlit as st
from PIL import Image

# Load the API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load Google Gemini Pro Model and get response
def get_response(prompt, input_text):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([prompt, input_text])
    return response.text

# Initialize the Streamlit app
st.title("Chavera MedBot: Your Medical Assistant")

# Creating radio section choices
section_choice = st.radio(
    "Choose Section:", 
    ("General Health Issues", "Critical Diseases", "General Medical Information", "Dietary Recommendations")
)

###########################################################################################
# If the choice is General Health Issues
if section_choice == "General Health Issues":

    # Prompt Template for General Health Issues
    input_prompt_health = """
    You are a medical assistant specializing in general health. Provide advice based on symptoms or conditions provided.
    Answer the following questions:
    - What could be the possible causes of the symptoms?
    - What are some recommended over-the-counter treatments or home remedies?
    - When should a patient consider seeing a doctor?
    Return the response using markdown.
    """
    
    # Input
    input_health = st.text_area("Describe your symptoms or general health issue:")
    # Button
    submit_health = st.button("Get Health Advice")
    if submit_health:
        response = get_response(input_prompt_health, input_health)
        st.subheader("Health Advisor Bot: ")
        st.write(response)

###########################################################################################
# If the choice is Critical Diseases
if section_choice == "Critical Diseases":

    # Prompt Template for Critical Diseases
    input_prompt_critical = """
    You are an expert in diagnosing and providing advice on critical diseases. Based on the information provided,
    answer the following questions:
    - What is the disease likely to be based on symptoms or conditions provided?
    - What are the common treatments or medications for this disease?
    - What lifestyle changes or follow-up steps are recommended?
    Return the response using markdown.
    """
    
    # Input
    input_critical = st.text_area("Describe the symptoms or medical condition:")
    # Button
    submit_critical = st.button("Get Diagnosis & Advice")
    if submit_critical:
        response = get_response(input_prompt_critical, input_critical)
        st.subheader("Critical Disease Bot: ")
        st.write(response)

###########################################################################################
# If the choice is General Medical Information
if section_choice == "General Medical Information":

    # Prompt Template for General Medical Information
    input_prompt_info = """
    You are a knowledgeable medical assistant providing general medical information. Answer the following:
    - Explain the medical condition, procedure, or term in simple language.
    - What are its causes, symptoms, and treatments?
    - Provide any additional important information related to the topic.
    Return the response using markdown.
    """
    
    # Input
    input_info = st.text_area("Ask about any medical condition, term, or procedure:")
    # Button
    submit_info = st.button("Get Medical Information")
    if submit_info:
        response = get_response(input_prompt_info, input_info)
        st.subheader("Medical Info Bot: ")
        st.write(response)

###########################################################################################
# If the choice is Dietary Recommendations
if section_choice == "Dietary Recommendations":

    # Prompt Template for Dietary Recommendations
    input_prompt_diet = """
    You are a dietician providing dietary recommendations. Based on the user’s input, provide:
    - General dietary advice for the mentioned condition, age, or lifestyle.
    - List specific foods to include or avoid.
    - Provide tips for maintaining a balanced diet.
    Return the response using markdown.
    """
    
    # Input
    input_diet = st.text_area("Enter your condition, age, or dietary preferences:")
    # Button
    submit_diet = st.button("Get Dietary Advice")
    if submit_diet:
        response = get_response(input_prompt_diet, input_diet)
        st.subheader("Dietary Advisor Bot: ")
        st.write(response)

###########################################################################################

# Clear response button
if st.button("Clear Response"):
    st.write("Response cleared.")
    st.text_area("")

