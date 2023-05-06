import pandas as pd
import pickle

class HealthInsurance( object ):
    
    def __init__(self):
        self.region_code_path = 'parameter/target_encoding_region_code.csv'
        self.policy_sales_path = 'parameter/frequency_encoding_policy_sales.csv'
        self.age_mms = pickle.load(open('parameter/mms_age.pkl', 'rb'))
        self.vintage_mms = pickle.load(open('parameter/mms_vintage.pkl','rb'))
        self.annual_premium_ss = pickle.load(open('parameter/ss_annual_premium.pkl', 'rb'))
         
        
    def preprocessing(self, df):

        X = df.copy()
        # Age
        X['age'] = self.age_mms.transform(X[['age']].values )        
           
        # Vintage
        X['vintage'] = self.vintage_mms.transform(X[['vintage']].values )

        # Annual premium
        X['annual_premium'] = self.annual_premium_ss.transform(X[['annual_premium']].values )        
 
        # Vehicle damage
        X = df.copy()
        X['vehicle_damage'] = X['vehicle_damage'].apply(lambda x: 0 if x == 'No' else 1)

        # region_code 
        target_encoding_region_code = pd.read_csv(self.region_code_path).set_index('region_code')['response']
        X['region_code'] = X['region_code'].map( target_encoding_region_code )
        X['region_code'].fillna(0, inplace = True)

        # policy sales channel
        frequency_encoding_policy_sales = pd.read_csv(self.policy_sales_path).set_index('policy_sales_channel')['response']
        X['policy_sales_channel'] = X['policy_sales_channel'].map( frequency_encoding_policy_sales )
        X['policy_sales_channel'].fillna(0, inplace = True)

        # Feature selection
        X = X[['age', 'region_code', 'policy_sales_channel', 'previously_insured', 'annual_premium', 'vintage', 'vehicle_damage']]  
        return X 
        
    def get_prediction( self, model, original_data, processed_data):
        predictions = model.predict_proba(processed_data)[::,1]
        
        # join predictions into the original data
        original_data['predictions'] = predictions
        original_data.sort_values( 'predictions', ascending = False )
        return original_data
