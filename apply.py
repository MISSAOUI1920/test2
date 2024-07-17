import spacy
import pytextrank
import streamlit as st

# Load a spaCy model
nlp = spacy.load("en_core_web_sm")

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
