import os
import pickle
import pandas        as pd
from datetime        import datetime
from flask           import Flask, request, Response
from HealthInsurance import HealthInsurance

# Get model pkl
model = pickle.load(open('model/xgb_model.pkl','rb'))

# Initiate API
app = Flask(__name__)

@app.route( '/healthinsurance/predict', methods=['POST'] )


def health_insurance_predict():
    test_json = request.get_json()

    if test_json: #there is data

        # Load data
        if isinstance( test_json, dict ): # unique example
            raw_df = pd.DataFrame(test_json, index=[0])    
        else: # multiple examples
            
            raw_df = pd.DataFrame(test_json, columns = test_json[0].keys())          

        # Instantiate HealthInsurance class
        pipeline = HealthInsurance()


        # Preprocessing
        df1 = pipeline.preprocessing(raw_df)

        # Predictions
        predictions = pipeline.get_prediction(model = model, original_data = raw_df, processed_data = df1)

        return predictions.to_json(orient = 'records', date_format = 'iso')

    else:
        return Response ('{}', status = 200, mimetype = 'application/json')


#### Local #####

#if __name__ == '__main__':
#    app.run( '10.0.0.175')

#### Web #####
if __name__ == '__main__':
    port = os.environ.get('PORT',5000)
    app.run( host = '0.0.0.0', port = port)
