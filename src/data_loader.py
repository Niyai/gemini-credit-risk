# src/data_loader.py
import pandas as pd
from ucimlrepo import fetch_ucirepo 

def load_and_process_credit_data():
    """
    Fetches and processes the German Credit Risk dataset using the ucimlrepo package.
    """
    print("Fetching German Credit Risk dataset via ucimlrepo...")
    try:
        # Fetch dataset using its ID from the UCI repository
        statlog_german_credit_data = fetch_ucirepo(id=144) 
        
        # data (as pandas dataframes) 
        X = statlog_german_credit_data.data.features 
        y = statlog_german_credit_data.data.targets 

        # Combine features and target into a single DataFrame
        df = pd.concat([X, y], axis=1)
        
        # --- Data Cleaning and Mapping ---
        # The library provides descriptive column names, but the values are still coded.
        # We will map them to be more interpretable for the LLM.
        status_map = {'A11': '< 0 DM', 'A12': '0 <= ... < 200 DM', 'A13': '>= 200 DM', 'A14': 'no checking account'}
        history_map = {'A30': 'no credits taken', 'A31': 'all credits paid back duly', 'A32': 'existing credits paid back duly till now', 'A33': 'delay in paying off in the past', 'A34': 'critical account'}
        purpose_map = {'A40': 'car (new)', 'A41': 'car (used)', 'A42': 'furniture/equipment', 'A43': 'radio/television', 'A44': 'domestic appliances', 'A45': 'repairs', 'A46': 'education', 'A47': 'vacation', 'A48': 'retraining', 'A49': 'business', 'A410': 'others'}
        sex_map = {'A91': 'male : divorced/separated', 'A92': 'female : divorced/separated/married', 'A93': 'male : single', 'A94': 'male : married/widowed', 'A95': 'female : single'}
        
        # Rename columns for clarity before mapping
        df.rename(columns={
            'Attribute1': 'Status of existing checking account',
            'Attribute2': 'Duration in month',
            'Attribute3': 'Credit history',
            'Attribute4': 'Purpose',
            'Attribute5': 'Credit amount',
            'Attribute7': 'Present employment since',
            'Attribute8': 'Installment rate in percentage of disposable income',
            'Attribute9': 'Personal status and sex',
            'Attribute13': 'Age',
            'Attribute15': 'Housing',
            'Attribute17': 'Job',
            'class': 'Risk' # The target variable is named 'class' by the library
        }, inplace=True)

        df['Status of existing checking account'] = df['Status of existing checking account'].map(status_map)
        df['Credit history'] = df['Credit history'].map(history_map)
        df['Purpose'] = df['Purpose'].map(purpose_map)
        df['Personal status and sex'] = df['Personal status and sex'].map(sex_map)

        # Create a separate 'Sex' column for fairness analysis
        df['Sex'] = df['Personal status and sex'].apply(lambda x: 'male' if 'male' in x else 'female')

        # Convert the target variable 'Risk' to a more intuitive format (1 = Bad, 0 = Good)
        df['Risk'] = df['Risk'].replace({1: 0, 2: 1})
        
        print("✅ German Credit dataset processed successfully.")
        return df

    except Exception as e:
        print(f"🔥 Error fetching or processing the data: {e}")
        return None
