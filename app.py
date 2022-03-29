from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("cancer.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict(final)
    output=prediction[0]
    return render_template('cancer.html',pred='{}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)