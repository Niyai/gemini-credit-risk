# data_preparation_for_finetuning.py
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

from src.data_loader import load_and_process_credit_data
from src.prompts import create_debiased_llm_prompt

def prepare_finetuning_dataset():
    """
    Prepares the processed real-world data for supervised fine-tuning by creating
    training and validation JSONL files in the required Gemini format.
    """
    print("Preparing real-world data for fine-tuning...")

    # Load the clean, customer-level data
    customer_df = load_and_process_credit_data()
    if customer_df is None:
        return

    # The ground truth is the 'isdelinquent' column (0=Good, 1=Bad)
    customer_df['ideal_output'] = customer_df['isdelinquent'].apply(lambda x: 'Bad' if x == 1 else 'Good')

    # --- Overfitting Mitigation: Split data into training and validation sets ---
    train_df, val_df = train_test_split(customer_df, test_size=0.2, random_state=42, stratify=customer_df['isdelinquent'])

    # --- FIX: Cap the validation set at 5000 records as required by the API ---
    if len(val_df) > 5000:
        val_df = val_df.head(5000)
        print("Validation dataset was larger than 5000 records. It has been capped.")
    # -------------------------------------------------------------------------

    # --- Create the JSONL files ---
    for dataset_type, df_split in [('train', train_df), ('validation', val_df)]:
        output_file_path = f'finetuning_dataset_{dataset_type}.jsonl'
        
        with open(output_file_path, 'w') as f:
            for index, row in df_split.iterrows():
                # The input will be our debiased prompt structure
                input_prompt = create_debiased_llm_prompt(row.to_dict())
                
                # The output is the ground truth from the dataset
                output_text = row['ideal_output']
                
                # --- CORRECTED FORMAT FOR GEMINI SFT ---
                # The API expects a 'contents' field with a list of roles and parts.
                json_record = {
                    "contents": [
                        {"role": "user", "parts": [{"text": input_prompt}]},
                        {"role": "model", "parts": [{"text": output_text}]}
                    ]
                }
                # ----------------------------------------
                
                f.write(json.dumps(json_record) + "\n")

        print(f"✅ {dataset_type.capitalize()} dataset successfully created at: {output_file_path} ({len(df_split)} records)")

if __name__ == "__main__":
    prepare_finetuning_dataset()
