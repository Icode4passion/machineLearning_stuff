# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:29:50 2018

@author: usunkesu
"""

from flask import Flask , jsonify , request
import pandas as pd

import traceback
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/predict',methods=['POST','GET'])
def predict():
    if lr:
        try:
                       
            json_ = request.json
            query_df = pd.DataFrame(json_)
            query = pd.get_dummies(query_df)
            query = query.reindex(columns=model_columns, fill_value=0)
            prediction = lr.predict(query)            
            return jsonify({'prediction':str(prediction)})
        
        except:
            
            return jsonify({'trace':traceback.format_exc()})
        
    else :
        print('Train model first')
        print('No Model here')



if __name__ == '__main__':
    lr = joblib.load('model.pkl')
    print('Model loaded')
    model_columns = joblib.load('model_columns.pkl')
    print("Model Columns loaded")
    
    app.run(debug=True)






















 