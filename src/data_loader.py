# src/data_loader.py
import pandas as pd
import numpy as np
from datetime import datetime

def clean_column_names(df):
    """Standardizes column names to be Python-friendly."""
    cols = df.columns
    new_cols = []
    for col in cols:
        new_col = col.strip().replace(' ', '_').replace('/', '_').replace('-', '_')
        new_col = ''.join(e for e in new_col if e.isalnum() or e == '_')
        new_cols.append(new_col.lower())
    df.columns = new_cols
    return df

def categorize_delinquency(days):
    """Categorizes the severity of delinquency based on days in arrears."""
    if days == 0:
        return 'No Arrears'
    elif 1 <= days <= 30:
        return '1-30 Days'
    elif 31 <= days <= 90:
        return '31-90 Days'
    else: # 91+ days
        return '90+ Days'

def load_and_process_credit_data(file_path="data/Credit_bureau_submission.xlsx"):
    """
    Loads and processes the real-world credit data from the specified Excel sheet,
    engineering features for analysis based on the refined project scope.
    """
    print("Processing real-world credit data from Excel file...")
    try:
        # Use pd.read_excel for .xlsx files and specify the sheet name
        df = pd.read_excel(file_path, sheet_name="Credit Information October ")
    except FileNotFoundError:
        print(f"🔥 Error: The file at {file_path} was not found.")
        return None
    except ValueError as e:
        print(f"🔥 Error: Could not find the sheet 'Credit Information October'. Please check the sheet name. Details: {e}")
        return None

    # 1. Standardize column names
    df = clean_column_names(df)

    # 2. Clean and convert data types
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
    numeric_cols = ['credit_limit_facility_amount_global_limit', 'outstanding_balance', 'days_in_arrears']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 3. Engineer Features at the loan level
    df['isdelinquent'] = (df['days_in_arrears'] > 0).astype(int)
    df['delinquencyseverity'] = df['days_in_arrears'].apply(categorize_delinquency)
    df['creditutilization'] = (df['outstanding_balance'] / df['credit_limit_facility_amount_global_limit']).replace([np.inf, -np.inf], 0).fillna(0)
    df['age'] = (datetime.now() - df['date_of_birth']).dt.days // 365
    
    # 4. Clean demographic features
    demographic_cols = ['gender', 'marital_status', 'primary_state']
    for col in demographic_cols:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)

    df['gender'] = df['gender'].str.strip().str.upper().replace({'M': 'Male', 'F': 'Female'})
    df['marital_status'] = df['marital_status'].str.strip().str.upper().replace({'S': 'Single', 'M': 'Married'})
    if 'primary_state' in df.columns:
        df['primary_state'] = df['primary_state'].str.replace(' State', '', case=False).str.strip()

    # 5. Aggregate to Customer Level
    customer_df = df.groupby('customerid').agg(
        age=('age', 'first'),
        gender=('gender', 'first'),
        primary_state=('primary_state', 'first'),
        marital_status=('marital_status', 'first'),
        total_outstanding=('outstanding_balance', 'sum'),
        average_utilization=('creditutilization', 'mean'),
        max_days_in_arrears=('days_in_arrears', 'max'),
        isdelinquent=('isdelinquent', 'max') 
    ).reset_index()

    # --- FIX: Fill any remaining NaN values in the age column with the median ---
    if customer_df['age'].isnull().any():
        median_age = customer_df['age'].median()
        customer_df['age'].fillna(median_age, inplace=True)
    # -------------------------------------------------------------------------

    customer_df['maxdelinquencyseverity'] = customer_df['max_days_in_arrears'].apply(categorize_delinquency)

    print(f"✅ Data processing complete. {len(customer_df)} unique customers found.")
    return customer_df
