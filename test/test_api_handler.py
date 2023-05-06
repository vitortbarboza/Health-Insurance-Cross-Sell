from   datetime  import datetime
import pandas    as pd
import requests
import json

df = pd.read_csv('../data/insurance_data.csv')
df = df.drop('response', axis = 1)
df = df.sample(20)

df = df.to_dict(orient = 'records')

json_df = json.dumps(df)

def predict(data):
    # API Call

    # Web URL
    url =  'https://healthinsurance-webapp-api.onrender.com/healthinsurance/predict'  

    # Local URL
    #url = 'http://10.0.0.175:5000/healthinsurance/

    header = {'Content-type': 'application/json' }
    data = data
    r = requests.post( url, data = data, headers = header )
    print( 'Status Code {}'.format( r.status_code ) )
    d1 = pd.DataFrame(r.json(), columns = r.json()[0].keys())
    d1.sort_values('predictions', ascending = False)
    return d1

predictions = predict(json_df)

# Export data
predictions.to_csv('../data/predictions/predictions-' + datetime.now().strftime("%d-%m-%Y %H:%M:%S")+'.csv', index = False)
