from flask import Flask,render_template,request

import FeatureExtraction
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/getURL',methods=['GET','POST'])
def getURL():
    if request.method == 'POST':
        url = request.form['url']
        print(url)
        data = FeatureExtraction.getAttributess(url)
        print(data)
        rf_model = pickle.load(open('RandomForestModel.sav', 'rb'))
        predicted_value = rf_model.predict(data)
        #print(predicted_value)
        if predicted_value == 0:    
            value = "This is a legitimate URL"
            return render_template("home.html", error=value)
        else:
            value = "This URL is suspected to be a phishing URL"
            return render_template("home.html", error=value)
if __name__ == "__main__":
    app.run(debug=True)