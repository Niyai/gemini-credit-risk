# src/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_data_for_woe(df, target_variable):
    """Prepares the dataset for WoE calculation."""
    # This now assumes 'totalNoOfDelinquent Facilities' has been created
    df['default'] = np.where(df['totalNoOfDelinquent Facilities'] > 0, 1, 0)

    # Impute missing values
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    return df

def calculate_woe_iv(df, feature, target):
    """Calculates WoE and IV for a single feature."""
    lst = []
    unique_vals = df[feature].unique()
    for val in unique_vals:
        all_count = df[df[feature] == val].shape[0]
        good_count = df[(df[feature] == val) & (df[target] == 0)].shape[0]
        bad_count = df[(df[feature] == val) & (df[target] == 1)].shape[0]
        lst.append({'Value': val, 'All': all_count, 'Good': good_count, 'Bad': bad_count})
        
    d2 = pd.DataFrame(lst)
    d2['Distr_Good'] = d2['Good'] / (d2['Good'].sum() + 1e-6)
    d2['Distr_Bad'] = d2['Bad'] / (d2['Bad'].sum() + 1e-6)
    d2['WoE'] = np.log((d2['Distr_Good'] + 1e-6) / (d2['Distr_Bad'] + 1e-6))
    d2 = d2.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    d2['IV'] = (d2['Distr_Good'] - d2['Distr_Bad']) * d2['WoE']
    
    iv = d2['IV'].sum()
    woe_map = dict(zip(d2.Value, d2.WoE))
    
    return woe_map, iv

class WoETransformer:
    """A transformer to apply WoE to the dataset."""
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.woe_maps = {}
        self.iv_scores = {}
        self.bin_edges = {}

    def fit(self, X, y):
        df = pd.concat([X, y], axis=1)
        target_name = y.name
        
        for feature in self.feature_names:
            if df[feature].nunique() > 10 and pd.api.types.is_numeric_dtype(df[feature]):
                try:
                    df[feature], self.bin_edges[feature] = pd.qcut(df[feature], q=4, retbins=True, duplicates='drop')
                except ValueError:
                    self.bin_edges[feature] = None
                    df[feature] = df[feature].astype(str)
            else:
                 self.bin_edges[feature] = None
                 df[feature] = df[feature].astype(str)

            woe_map, iv = calculate_woe_iv(df, feature, target_name)
            self.woe_maps[feature] = woe_map
            self.iv_scores[feature] = iv
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for feature in self.feature_names:
            if self.bin_edges.get(feature) is not None:
                X_transformed[feature] = pd.cut(X_transformed[feature], bins=self.bin_edges[feature], include_lowest=True)
            
            X_transformed[feature] = X_transformed[feature].astype(str)
            woe_map = self.woe_maps[feature]
            X_transformed[feature] = X_transformed[feature].map(woe_map).fillna(0)
        return X_transformed
