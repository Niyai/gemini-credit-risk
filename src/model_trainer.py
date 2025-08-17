# src/model_trainer.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump

def train_and_save_model(df: pd.DataFrame):
    """
    Trains a benchmark Logistic Regression model on the German Credit dataset.
    """
    print("Training benchmark model on German Credit data...")
    
    # Define features and target based on the German dataset
    # We select a mix of numeric and categorical features
    features = [
        'Duration in month', 'Credit amount', 'Installment rate in percentage of disposable income',
        'Age', 'Credit history', 'Purpose', 'Housing', 'Job'
    ]
    target = 'Risk'

    X = df[features]
    y = df[target]

    # Define preprocessing for categorical and numeric features
    categorical_features = ['Credit history', 'Purpose', 'Housing', 'Job']
    numeric_features = ['Duration in month', 'Credit amount', 'Installment rate in percentage of disposable income', 'Age']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Create the full pipeline with preprocessing and the model
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))])
    
    # Split data and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    model_pipeline.fit(X_train, y_train)

    # Save the trained pipeline
    dump(model_pipeline, 'credit_model_german.joblib')

    print("✅ Benchmark model for German data saved successfully.")
