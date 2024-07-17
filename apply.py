import spacy
import pytextrank
import streamlit as st
import requests
import tarfile
import os
from pathlib import Path

# Function to download the spaCy model
def download_spacy_model():
    url = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz"
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open("en_core_web_sm.tar.gz", "wb") as f:
            f.write(response.raw.read())

        with tarfile.open("en_core_web_sm.tar.gz", "r:gz") as tar:
            tar.extractall()
        
        model_path = Path("en_core_web_sm-3.5.0/en_core_web_sm/en_core_web_sm-3.5.0")
        return model_path
    else:
        raise Exception("Failed to download spaCy model")

# Check if the model is already downloaded
model_path = Path("en_core_web_sm-3.5.0/en_core_web_sm/en_core_web_sm-3.5.0")

if not model_path.exists():
    model_path = download_spacy_model()

# Load the spaCy model
nlp = spacy.load(model_path)

# Add PyTextRank to the spaCy pipeline
nlp.add_pipe("textrank", last=True)

# Streamlit app
st.title("Key Phrase Extraction with PyTextRank")

# Input text
input_text = st.text_input("Enter text:", "i want solutions for leaf spot ?")

# Process the input text when the input is not empty
if input_text:
    doc = nlp(input_text)
    # Extract the top key phrase
    top_phrase = doc._.phrases[0].text
    # Display the top key phrase
    st.write("Top key phrase:", top_phrase)
