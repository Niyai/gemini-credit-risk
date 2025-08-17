# src/risk_scorer.py
from joblib import load
import pandas as pd
import os

class MLRiskScorer:
    """
    Loads the pre-trained ML model to score applicants from the German dataset.
    """
    def __init__(self, model_path='credit_model_german.joblib'):
        self.model_pipeline = None
        if os.path.exists(model_path):
            self.model_pipeline = load(model_path)
        else:
            print("Warning: ML model for German data not found. Please train first.")

    def predict_risk(self, customer_data: dict) -> str:
        """
        Predicts the risk ('Good' or 'Bad') for an applicant using the trained ML model.
        """
        if self.model_pipeline is None:
            return "Error"

        # Prepare the input data in a DataFrame
        df = pd.DataFrame([customer_data])
        
        # Predict the class (0 for Good, 1 for Bad)
        prediction = self.model_pipeline.predict(df)[0]
        
        return 'Bad' if prediction == 1 else 'Good'
