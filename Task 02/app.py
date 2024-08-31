import streamlit as st
import re
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model and tokenizer
model = load_model('fake_news_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Streamlit app UI
st.title("Fake Text Detection")

input_text = st.text_area("Enter the text:")

if st.button("Predict"):
    # Preprocess the input
    cleaned_text = clean_text(input_text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    max_length = 200
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

    # Predict
    prediction = (model.predict(padded_sequence) > 0.5).astype("int32")

    # Output the prediction
    if prediction == 0:
        st.write("The text is predicted to be **FAKE**.")
    else:
        st.write("The text is predicted to be **REAL**.")
