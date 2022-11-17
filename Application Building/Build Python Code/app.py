
import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app=Flask(__name__, template_folder='templates')
model=pickle.load(open('CKD.pkl','rb'))


@app.route('/Home.html')
def Home():
    return render_template('Home.html')           


@app.route('/prediction page.html')
def prediction():
    return render_template('prediction page.html')



@app.route('/predict',methods=['POST','GET'])
def predict():
    
    input_features=[float(x) for x in request.form.values()]
    features_value=[np.array(input_features)]

    features_name=['blood_urea','blood glucose random','coronary_artery_disease','anemia','pus_cell','red_blood_cells','diabetesmellitus','pedal_edema']
    
    df=pd.DataFrame(features_value, columns=features_name)

    pred=model.predict(df)
    
    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True) 