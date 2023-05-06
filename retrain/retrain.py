import pickle
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from xgboost  import XGBClassifier
from sqlalchemy              import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.preprocessing   import MinMaxScaler
from sqlalchemy.sql          import text
from sqlalchemy              import create_engine

####### Helper Functions #########

def score_top_k(y_test, predict_proba, top_k):
    df = pd.DataFrame()
    df['predictions'] = predict_proba[:,1]
    df['y_test'] = y_test.values
    df = df.sort_values('predictions', ascending = False).reset_index(drop = True).reset_index()
    df['score'] = df.apply(lambda x: 1 if ( x['index'] <= top_k and x['y_test'] == 1) else 0, axis = 1 )
    precision = df['score'].sum() / top_k
    recall = df['score'].sum() / df['y_test'].sum()
    return {'precision': precision, 'recall': recall}


####### Engine to connect database ##########
engine = create_engine('postgresql+psycopg2://member:cdspa@comunidade-ds-postgres.c50pcakiuwi3.us-east-1.rds.amazonaws.com/comunidadedsdb')
conn = engine.connect()

query = '''
        SELECT 
              pu.id,
              gender,
              age,
              region_code,
              policy_sales_channel,
              previously_insured,
              annual_premium,
              vintage,
              response,
              driving_license,
              vehicle_age,
              vehicle_damage
            
        FROM
              pa004.users pu
              LEFT JOIN pa004.insurance pi ON (pi.id = pu.id)
              LEFT JOIN pa004.vehicle pv ON (pi.id = pv.id);
'''

with conn.execution_options(autocommit=True) as conn:
    query = conn.execute(text(query))
   
df = pd.DataFrame(query.fetchall())
df.columns = ['id','gender','age','region_code','policy_sales_channel','previously_insured',
              'annual_premium','vintage','response','driving_license','vehicle_age','vehicle_damage']

conn.close()

X = df.drop('response', axis = 1)
y = df['response']


# Train test split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2)

######## Preprocessing ##########

# Age
mms_age = MinMaxScaler()
X_train['age'] = mms_age.fit_transform(X_train[['age']].values )
pickle.dump( mms_age, open( '../src/parameter/mms_age.pkl', 'wb' ) )
X_test['age'] = mms_age.transform(X_test[['age']].values )

# Vintage
mms_vintage = MinMaxScaler()
X_train['vintage'] = mms_vintage.fit_transform(X_train[['vintage']].values )
pickle.dump( mms_vintage, open( '../src/parameter/mms_vintage.pkl', 'wb' ) )
X_test['vintage'] = mms_vintage.transform(X_test[['vintage']].values )

# Annual premium
ss_annual_premium = StandardScaler()
X_train['annual_premium'] = ss_annual_premium.fit_transform(X_train[['annual_premium']].values )
pickle.dump( ss_annual_premium, open( '../src/parameter/ss_annual_premium.pkl', 'wb' ) )
X_test['annual_premium'] = ss_annual_premium.transform(X_test[['annual_premium']].values )

# Vehicle damage
X_train['vehicle_damage'] = X_train['vehicle_damage'].apply(lambda x: 0 if x == 'No' else 1)
X_test['vehicle_damage'] = X_test['vehicle_damage'].apply(lambda x: 0 if x == 'No' else 1)

# region_code 
aux = X_train.copy()
aux['response'] = y
target_encoding_region_code = aux.groupby('region_code')['response'].mean()
target_encoding_region_code.to_csv('../src/parameter/target_encoding_region_code.csv', index = True )
X_train['region_code'] = X_train['region_code'].map( target_encoding_region_code )
X_train['region_code'].fillna(0, inplace = True)
X_test['region_code'] = X_test['region_code'].map( target_encoding_region_code )
X_test['region_code'].fillna(0, inplace = True)

# Gender
X_train['gender'] = X_train['gender'].apply(lambda x: 0 if x == 'Male' else 1)
X_test['gender'] = X_test['gender'].apply(lambda x: 0 if x == 'Male' else 1)  
  
# Vehicle age
X_train['vehicle_age'] = X_train['vehicle_age'].apply(lambda x: 0 if x == '< 1 Year' else 1 if x == '1-2 Year' else 2)  
X_test['vehicle_age'] = X_test['vehicle_age'].apply(lambda x: 0 if x == '< 1 Year' else 1 if x == '1-2 Year' else 2)  

# policy sales channel
frequency_encoding_policy_sales = aux.groupby('policy_sales_channel')['response'].count() / len(X)
frequency_encoding_policy_sales.to_csv('../src/parameter/frequency_encoding_policy_sales.csv', index = True )
X_train['policy_sales_channel'] = X_train['policy_sales_channel'].map( frequency_encoding_policy_sales )
X_train['policy_sales_channel'].fillna(0, inplace = True)
X_test['policy_sales_channel'] = X_test['policy_sales_channel'].map( frequency_encoding_policy_sales )
X_test['policy_sales_channel'].fillna(0, inplace = True)

######## Feature selection ##########
X_train = X_train[['age', 'region_code', 'policy_sales_channel', 'previously_insured', 'annual_premium', 'vintage', 'vehicle_damage']]
X_test = X_test[['age', 'region_code', 'policy_sales_channel', 'previously_insured', 'annual_premium', 'vintage', 'vehicle_damage']]

######## Create ML Model ##########

params = {'max_depth': 8,
         'learning_rate': 0.014416504517463456,
         'n_estimators': 499,
         'min_child_weight': 10,
         'scale_pos_weight': 10,
         'gamma': 0.5834982442488319,
         'subsample': 0.3703846362615623,
         'colsample_bytree': 0.6808135028844782,
         'reg_alpha': 1.001572471362163e-05,
         'reg_lambda': 1.1690461872764534e-05}

########## Train model #############
xgb_model = XGBClassifier(**params).fit(X_train, y_train)

# Save model pkl
pickle.dump( xgb_model, open( '../src/model/xgb_model.pkl', 'wb' ) )

top_k = 2000

######### Evaluate model ###########
results = score_top_k(y_test, xgb_model.predict_proba(X_test), top_k)

date = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
results = [top_k, results['precision'], results['recall'], date]

df_results = pd.DataFrame(results).T
df_results.columns = ['top_k','precision', 'recall', 'datetime']

######### Save results ############

# create table
query_create_table = '''
        CREATE TABLE results (
                                top_k     INTEGER,
                                precision REAL,
                                recall    REAL,
                                datetime  TEXT
        )
'''

# Connect to database and create table
#conn = sqlite3.connect( 'results/results_db.sqlite' )
#cursor = conn.execute( query_create_table )
#conn.commit()
#conn.close()

# Insert data into table
conn = sqlite3.connect( 'results/results_db.sqlite' )
df_results.to_sql( 'results', con = conn, if_exists = 'append', index = False)
conn.close()
