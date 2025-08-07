# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 23:44:43 2025

@author: THINKPAD
"""




import streamlit as st
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize


def preprocess(text):
    text = text.lower() 
    text = text.translate(str.maketrans("", "", string.punctuation)) 
    return text


with open("emotions.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()


sentences = sent_tokenize(raw_text)


def get_most_relevant_sentence(user_query):
    processed_sentences = [preprocess(s) for s in sentences]
    processed_query = preprocess(user_query)
    
    all_sentences = [processed_query] + processed_sentences
    
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    vectors = vectorizer.fit_transform(all_sentences)
    
    similarity_scores = cosine_similarity(vectors[0:1], vectors[1:])
    most_similar_index = similarity_scores.argsort()[0][-1]
    
    return sentences[most_similar_index]


def chatbot(user_query):
    return get_most_relevant_sentence(user_query)


def main():
    st.title("Emotions Chatbot ")
    st.write("Ask me anything about emotions!")

    user_input = st.text_input("You:")
    if user_input:
        answer = chatbot(user_input)
        st.write("Bot:", answer)

if __name__ == "__main__":
    main()