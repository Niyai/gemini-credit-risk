# data_preparation_for_finetuning.py
import pandas as pd
import json

from src.data_loader import load_and_process_credit_data
from src.prompts import create_debiased_llm_prompt

def prepare_finetuning_dataset():
    """
    Prepares the German Credit dataset for supervised fine-tuning using the
    modern Gemini conversation format with the required 'contents' and 'parts' fields.
    """
    print("Preparing German credit data for fine-tuning...")

    credit_df = load_and_process_credit_data()
    if credit_df is None:
        return

    # The ground truth is the 'Risk' column (0=Good, 1=Bad)
    credit_df['ideal_output'] = credit_df['Risk'].apply(lambda x: 'Bad' if x == 1 else 'Good')

    output_file_path = 'finetuning_dataset_german.jsonl'

    with open(output_file_path, 'w') as f:
        for index, row in credit_df.iterrows():
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

    print(f"✅ Fine-tuning dataset successfully created at: {output_file_path}")
    print(f"Total records prepared: {len(credit_df)}")

if __name__ == "__main__":
    prepare_finetuning_dataset()
