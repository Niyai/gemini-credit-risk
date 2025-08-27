# src/bias_analyzer.py
import pandas as pd
import random

def create_counterfactual(customer_data: pd.Series, bias_variable='age') -> pd.Series:
    """
    Creates a counterfactual version of the customer data by changing a sensitive attribute.
    
    Args:
        customer_data: A pandas Series representing a single customer.
        bias_variable: The sensitive attribute to change ('age', 'gender', or 'primary_state').

    Returns:
        A new pandas Series with the modified attribute.
    """
    counterfactual_data = customer_data.copy()
    
    if bias_variable == 'age':
        # Create a "younger" version of the applicant for the test
        counterfactual_data['age'] = 24
    
    elif bias_variable == 'gender':
        # Swap the gender
        current_gender = customer_data.get('gender', 'Unknown')
        if current_gender == 'Male':
            counterfactual_data['gender'] = 'Female'
        else:
            # If original is Female or Unknown, change to Male
            counterfactual_data['gender'] = 'Male'

    elif bias_variable == 'primary_state':
        # Change the state to a different, randomly selected major state
        # to test for geographic bias.
        current_state = customer_data.get('primary_state', 'Unknown')
        possible_states = ['Lagos', 'Kano', 'Rivers', 'Abuja FCT', 'Anambra', 'Oyo']
        
        # Ensure the new state is different from the original
        new_state_options = [s for s in possible_states if s != current_state]
        if new_state_options:
            counterfactual_data['primary_state'] = random.choice(new_state_options)
        else:
            # Fallback if the current state is the only one in our list
            counterfactual_data['primary_state'] = 'Lagos'
            
    return counterfactual_data
