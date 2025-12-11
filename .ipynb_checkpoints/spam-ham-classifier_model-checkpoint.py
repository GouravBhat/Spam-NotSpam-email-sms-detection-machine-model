import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import nltk
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
import string
import os
import joblib


MODEL_FILE="model.pkl"
TLI_DF="TLI_df.pkl"

def text_transform(text):
    ps=PorterStemmer()
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i  not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

if not os.path.exists(MODEL_FILE):
    # lets train the model
    Data = pd.read_csv("clean_spam.csv")
    encoder=LabelEncoder()
    Data['label']=encoder.fit_transform(Data['label'])
    print(encoder.classes_)
    Data['text_transform']=Data['message'].apply(text_transform) 
    tfidf=TfidfVectorizer()

    x=tfidf.fit_transform(Data['text_transform']).toarray() 

    y=Data['label'].values

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


    model= RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(x_train,y_train)
 
    joblib.dump(model, MODEL_FILE)
    joblib.dump(tfidf, TLI_DF)
    

   
    print("Model trained and saved.")
else:
    # interference done here
    print("Model already exists. Loading the model.")
    model = joblib.load(MODEL_FILE)
    tfidf=joblib.load(TLI_DF)
   
    trasform_text=text_transform(f"Dear user, your chat backup has been stopped due to low space.To continue end-to-end encrypted backup, verify your device:👉 https://wa-backup-verify.inReply ‘DONE’ after verification.")
    print(trasform_text)
    vector_input=tfidf.transform([trasform_text]).toarray()
    
    result=model.predict(vector_input)
    if result[0]==1:
        print("Spam")
    else:
        print("Ham")

    
    