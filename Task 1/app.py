import streamlit as st
import pandas as pd
import re
from nltk.tokenize import sent_tokenize
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Data preprocessing function
def preprocess_text(text):
    try:
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation
        sentences = sent_tokenize(text)  # Sentence tokenization
        return sentences
    except Exception as e:
        st.error(f"Error preprocessing text: {e}")
        return []

# Function to generate story
def generate_story(prompt, max_length=200, temperature=0.7, top_k=50, top_p=0.9):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=None,
        num_return_sequences=1,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    story = tokenizer.decode(output[0], skip_special_tokens=True)
    return story

# Function to generate dialogue
def generate_dialogue(prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.2):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=None,
        do_sample=True,
        repetition_penalty=repetition_penalty
    )
    dialogue = tokenizer.decode(output[0], skip_special_tokens=True)
    dialogue_lines = re.split(r'(?<=[.!?])\s+', dialogue)
    return dialogue_lines

# Streamlit app
st.title('Text Generation with GPT-2')

# Upload file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    text_data = df['message'].tolist()

    # Preprocess text
    preprocessed_stories = [preprocess_text(story) for story in text_data]
    
    # Generate story
    prompt = st.text_input("Enter a prompt for story generation", "In a distant galaxy, a brave warrior")
    if st.button("Generate Story"):
        story = generate_story(prompt)
        st.write("Generated Story:")
        st.write(story)
    
    # Generate dialogue
    dialogue_prompt = st.text_input("Enter a prompt for dialogue generation", "Hero: Where are we heading?")
    if st.button("Generate Dialogue"):
        dialogue = generate_dialogue(dialogue_prompt)
        st.write("Generated Dialogue:")
        for line in dialogue:
            st.write(line)
