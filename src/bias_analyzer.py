# src/bias_analyzer.py
import pandas as pd

def create_counterfactual(customer_data: pd.Series, bias_variable='Age') -> pd.Series:
    """
    Creates a counterfactual version of the customer data by changing a sensitive attribute.
    """
    counterfactual_data = customer_data.copy()
    
    if bias_variable == 'Age':
        # Create a "younger" version of the applicant
        counterfactual_data['Age'] = 25
    
    elif bias_variable == 'Sex':
        # Swap the sex
        counterfactual_data['Sex'] = 'female' if customer_data['Sex'] == 'male' else 'male'
            
    return counterfactual_data
