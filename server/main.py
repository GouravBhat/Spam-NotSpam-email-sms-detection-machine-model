from flask import Flask, request, jsonify
import joblib
from nltk.stem.porter import PorterStemmer
import nltk
import string
from flask_cors import CORS
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt_tab')

MODEL_FILE="model.pkl"
TLI_DF="TLI_df.pkl"
model = joblib.load(MODEL_FILE)
tfidf=joblib.load(TLI_DF)

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

app = Flask(__name__)
CORS(app)   # <-- FIX CORS
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    message = data.get("message")
    try:
        if(message.strip() == ""):
         return jsonify({'message': 'Empty message provided.please fill'}), 400
        else:
         trasform_text=text_transform(message)
         vector_input=tfidf.transform([trasform_text]).toarray()
         result=model.predict(vector_input)
         if result[0]==1:
                prediction="spam"
         else:
                prediction="not Spam"
         return jsonify({'data': prediction})
    
    except Exception as e:
        return jsonify({'message': str(e)}), 500


   
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
 