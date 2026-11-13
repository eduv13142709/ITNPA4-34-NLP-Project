import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import streamlit as st

st.title('Researchly Abstract Classifier')
st.write('This app classifies research abstract as either AI related or classical research')

#nltk.download('punkt', quiet=True)
#nltk.download('punkt_tab', quiet=True)
#nltk.download('stopwords', quiet=True)
#nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
model = joblib.load('researchly_classifier.pkl')

def preprocess_text(text):
    clean_text = re.sub(r'[^a-zA-Z\s]','',str(text)).lower()
    tokens = word_tokenize(clean_text)

    filtered_words = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words and len(w) > 1
    ]
    return ' '.join(filtered_words)

abstract_input = st.text_area('Enter research abstract', height=200)

if st.button('Classify Abstract'):
    if abstract_input.strip():
        cleaned = preprocess_text(abstract_input)
        prediction = model.predict([cleaned])

        st.subheader('Prediction')
        if prediction[0] == 1:
             st.success('The abstract is classifies as: artificial research')
        else:
            st.success('The abstract is classifies as: classical research')
    else:
        st.warning('Please eneter research abstract before classifying')


