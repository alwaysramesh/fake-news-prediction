from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("C:/Users/ALWAYSRAMESH/fake_news_detection/nb_model.pkl", "rb"))
vectorizer = pickle.load(open("C:/Users/ALWAYSRAMESH/fake_news_detection/count_vect_title_text.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news']
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)
        
        result = "🟢 Real News" if prediction[0] == 1 else "🔴 Fake News"
        return render_template('index.html', prediction=result, news=news_text)

if __name__ == "__main__":
    app.run()
