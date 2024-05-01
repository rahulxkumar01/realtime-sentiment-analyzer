import pickle
import nltk
import streamlit as st
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
    #tokens = [token for token in tokens if token not in stop_words]
    return " ".join(tokens)

clf = pickle.load(open('movie_review_classifierr.pkl', 'rb'))

vectorizer = pickle.load(open('movie_review_vectorizer.pkl', 'rb'))


def model(text):
  text = preprocess_text(text)
  text_vector = vectorizer.transform([text])
  prediction = clf.predict(text_vector)[0]
  return prediction
def main():
  html="""
  <div style="background-color:white; padding:10px;">
  <h1 style="color:Black; text-align:center;">Review Classifier</h1>
  </div>
  """
  st.markdown(html,unsafe_allow_html=True) 
  text= st.text_input("Write a review...")
  if st.button("Submit"):
    sentiment=model(text)
    if sentiment== "pos":
      st.success("Positive Review")
    else:
      st.success("Negative Review")
if __name__=="__main__":
  main()
