from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

load_model = None
with open('basic_classifier.pkl', 'rb') as fid:
    loaded_model = pickle.load(fid)

vectorizer = None
with open('count_vectorizer.pkl', 'fb') as vd:
    vectorizer = pickle.load(vd)

prediction = loaded_model.predict(vectorizer.transform(['This is fake news']))[0]