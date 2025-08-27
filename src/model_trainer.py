# src/model_trainer.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump

def train_and_save_model(df: pd.DataFrame):
    """
    Trains a benchmark Logistic Regression model on the processed real-world data.
    """
    print("Training benchmark model...")
    
    # Define features and the target variable from our processed DataFrame
    # We select a mix of the engineered numeric and cleaned categorical features
    numeric_features = ['age', 'total_outstanding', 'average_utilization', 'max_days_in_arrears']
    categorical_features = ['gender', 'primary_state', 'maxdelinquencyseverity']
    
    
    target = 'isdelinquent'

    # Ensure all required columns are present
    required_cols = numeric_features + categorical_features + [target]
    for col in required_cols:
        if col not in df.columns:
            print(f"🔥 Error: Required column '{col}' not found in the DataFrame.")
            return

    X = df[numeric_features + categorical_features]
    y = df[target]

    # Define preprocessing steps for numeric and categorical data
    # Numeric features are scaled to have a mean of 0 and variance of 1
    # Categorical features are converted into a numerical format using one-hot encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Create the full machine learning pipeline
    # This pipeline first preprocesses the data and then trains the classifier
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
    ])
    
    # Split data for training and testing, ensuring the class distribution is the same in both sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Save the entire trained pipeline for later use
    dump(model_pipeline, 'credit_model.joblib')

    print("✅ Benchmark model saved successfully.")
