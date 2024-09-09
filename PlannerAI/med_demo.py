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
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content([prompt, input_text])
    return response.text


# Function to load Google Gemini Vision Model and get response from images
def get_response_image(image, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([image[0], prompt])
    return response.text

# Prepare Image Data
#def prep_image(uploaded_file):
 #   if uploaded_file is not None:
  #      bytes_data = uploaded_file.getvalue()
   #     image_parts = [{"mime_type": uploaded_file.type, "data": bytes_data}]
    #    return image_parts
    #else:
     #   raise FileNotFoundError("No File is uploaded!")
    
def prep_image(uploaded_file):
    #Check if there is any data
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()

        #Get the image part information
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No File is uploaded!")
        

# Initialize the Streamlit app
st.title("Chavera MedBot: Your Medical Assistant")

# Creating radio section choices
section_choice = st.radio(
    "Choose Section:", 
    (
        "Upload Image for Diagnosis",
        "General Health Issues", 
        "Critical Diseases", 
        "General Medical Information", 
        "Dietary Recommendations",  
        "Neurologist", 
        "Dermatologist", 
        "Cardiologist", 
        "Obstetrics and Gynecology", 
        "Pediatrician"
    )
)


###########################################################################################
# Section for Upload Image for Diagnosis
if section_choice == "Upload Image for Diagnosis":
    input_prompt_image = """
    You are a highly experienced medical expert specializing in diagnosing medical conditions based on visual analysis of images. Carefully analyze the uploaded image and provide a detailed explanation.

    Your response should include:
    1. A description of any noticeable abnormalities, patterns, or features in the image.
    2. Possible medical conditions or diagnoses that could be associated with the visual findings.
    3. Recommended next steps, such as further medical tests, consultations, or treatments, if necessary.
    4. A brief overview of why these findings could be significant, including potential health impacts.

    Ensure the response is clear, concise, and suitable for a patient-friendly report. Use bullet points or numbered lists for clarity and return the response using markdown.
    """

    # Image uploader
    upload_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if upload_file is not None:
       image = Image.open(upload_file)
       st.image(image, caption="Uploaded Image", use_column_width=True)

    # Button to analyze the uploaded image
    submit_image = st.button("Analyze Image")

    if submit_image and upload_file is not None:
        with st.spinner("Analyzing..."):
            try:
               # Prepare the image data and get the response
               image_data = prep_image(upload_file)
               response = get_response_image(image_data, input_prompt_image)
               st.subheader("Image Diagnosis Bot: ")
               st.write(response)
            except Exception as e:
              st.error(f"An error occurred: {str(e)}")
    else:
        st.write("Please upload an image to analyze.")

###########################################################################################
# Section for General Health Issues
elif section_choice == "General Health Issues":
    input_prompt_health = """
    You are a medical assistant specializing in general health. Provide advice based on symptoms or conditions provided.
    Answer the following questions:
    - What could be the possible causes of the symptoms?
    - What are some recommended over-the-counter treatments or home remedies?
    - When should a patient consider seeing a doctor?
    Return the response using markdown.
    """
    
    input_health = st.text_area("Describe your symptoms or general health issue:")
    submit_health = st.button("Get Health Advice")
    if submit_health:
        response = get_response(input_prompt_health, input_health)
        st.subheader("Health Advisor Bot: ")
        st.write(response)

###########################################################################################
# Section for Critical Diseases
elif section_choice == "Critical Diseases":
    input_prompt_critical = """
    You are an expert in diagnosing and providing advice on critical diseases. Based on the symptoms or medical conditions provided, answer the following:
    - What is the most likely disease or condition?
    - What are the common treatments or medications for this disease?
    - What lifestyle changes or follow-up steps are recommended?
    Return the response using markdown.
    """
    input_critical = st.text_area("Describe the symptoms or medical condition:")
    submit_critical = st.button("Get Diagnosis & Advice")
    if submit_critical:
        response = get_response(input_prompt_critical, input_critical)
        st.subheader("Critical Disease Bot: ")
        st.write(response)

###########################################################################################
# Section for General Medical Information
elif section_choice == "General Medical Information":
    input_prompt_info = """
    You are an expert in diagnosing and providing advice on critical diseases. Based on the symptoms or medical conditions provided, answer the following:
    - What is the most likely disease or condition?
    - What are the common treatments or medications for this disease?
    - What lifestyle changes or follow-up steps are recommended?
    Return the response using markdown.
    """

    input_info = st.text_area("Ask about any medical condition, term, or procedure:")
    submit_info = st.button("Get Medical Information")
    if submit_info:
        response = get_response(input_prompt_info, input_info)
        st.subheader("Medical Info Bot: ")
        st.write(response)

###########################################################################################
# Section for Dietary Recommendations
elif section_choice == "Dietary Recommendations":
    input_prompt_diet = """
    You are a dietician providing dietary recommendations. Based on the user’s input, provide:
    - General dietary advice for the mentioned condition, age, or lifestyle.
    - List specific foods to include or avoid.
    - Provide tips for maintaining a balanced diet.
    Return the response using markdown.
    """
    input_diet = st.text_area("Enter your condition, age, or dietary preferences:")
    submit_diet = st.button("Get Dietary Advice")
    if submit_diet:
        response = get_response(input_prompt_diet, input_diet)
        st.subheader("Dietary Advisor Bot: ")
        st.write(response)



###########################################################################################
# Sections for Specialist Consultations (Neurologist, Dermatologist, etc.)
elif section_choice == "Neurologist":
    input_prompt_neuro = """
    You are a neurologist providing specialized advice for neurological symptoms. Based on the provided symptoms or concerns, answer the following:
    - What could be the possible neurological condition?
    - What are the recommended diagnostic tests or examinations?
    - What are the common treatments or management options?
    Return the response using markdown.
    """
    input_neuro = st.text_area("Describe your neurological symptoms or concerns:")
    submit_neuro = st.button("Get Neurology Advice")
    if submit_neuro:
        response = get_response(input_prompt_neuro, input_neuro)
        st.subheader("Neurologist Bot: ")
        st.write(response)

elif section_choice == "Dermatologist":
    input_prompt_derm = """
    You are a dermatologist providing advice on skin-related issues. Based on the provided symptoms or skin condition, answer the following:
    - What could be the possible skin condition?
    - What are the recommended treatments or skincare routines?
    - When should a patient seek professional medical care?
    Return the response using markdown.
    """
    input_derm = st.text_area("Describe your skin condition or symptoms:")
    submit_derm = st.button("Get Dermatology Advice")
    if submit_derm:
        response = get_response(input_prompt_derm, input_derm)
        st.subheader("Dermatologist Bot: ")
        st.write(response)

elif section_choice == "Cardiologist":
    input_prompt_cardio = """
    You are a cardiologist providing advice on heart-related concerns. Based on the provided symptoms or conditions, answer the following:
    - What could be the potential heart condition or concern?
    - What are the common diagnostic tests and treatments?
    - What lifestyle changes are recommended to manage or prevent this condition?
    Return the response using markdown.
    """
    input_cardio = st.text_area("Describe your heart-related symptoms or conditions:")
    submit_cardio = st.button("Get Cardiology Advice")
    if submit_cardio:
        response = get_response(input_prompt_cardio, input_cardio)
        st.subheader("Cardiologist Bot: ")
        st.write(response)

elif section_choice == "Obstetrics and Gynecology":
    input_prompt_obgyn = """
    You are an obstetrician/gynecologist providing advice on reproductive health. Based on the patient's symptoms or questions, answer the following:
    - What could be the potential reproductive health concern or condition?
    - What are the common diagnostic methods and treatments?
    - What are the important follow-up steps or precautions to take?
    Return the response using markdown.
    """
    input_obgyn = st.text_area("Describe your reproductive health concerns or questions:")
    submit_obgyn = st.button("Get OB/GYN Advice")
    if submit_obgyn:
        response = get_response(input_prompt_obgyn, input_obgyn)
        st.subheader("OB/GYN Bot: ")
        st.write(response)

elif section_choice == "Pediatrician":
    input_prompt_pedi = """
    You are a pediatrician providing advice on child health issues. Based on the child's symptoms or health concerns, answer the following:
    - What could be the possible health issue?
    - What are the recommended treatments or home care practices?
    - When should a parent seek further medical attention for their child?
    Return the response using markdown.
    """
    input_pedi = st.text_area("Describe the child's symptoms or health concerns:")
    submit_pedi = st.button("Get Pediatric Advice")
    if submit_pedi:
        response = get_response(input_prompt_pedi, input_pedi)
        st.subheader("Pediatrician Bot: ")
        st.write(response)

###########################################################################################
# Button to clear response
if st.button("Clear Response"):
    st.session_state['response'] = ""
    st.write("Response cleared.")
