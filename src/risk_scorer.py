# src/risk_scorer.py
import pandas as pd
from joblib import load
import os

class MLRiskScorer:
    """
    Loads the pre-trained ML model to predict delinquency risk for new applicants.
    """
    def __init__(self, model_path='credit_model.joblib'):
        self.model_pipeline = None
        if os.path.exists(model_path):
            self.model_pipeline = load(model_path)
        else:
            print(f"🔥 Warning: Benchmark model not found at {model_path}. Please run the training process first.")

    def predict_risk(self, customer_data: dict) -> str:
        """
        Predicts the risk ('Good' or 'Bad') for an applicant using the trained ML model.
        """
        if self.model_pipeline is None:
            return "Error: Model not loaded"

        # Prepare the input data in a DataFrame, ensuring the order of columns matches training
        df = pd.DataFrame([customer_data])
        
        # Predict the class (0 for Good/Not Delinquent, 1 for Bad/Delinquent)
        prediction = self.model_pipeline.predict(df)[0]
        
        return 'Bad' if prediction == 1 else 'Good'
