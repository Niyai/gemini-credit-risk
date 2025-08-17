# test_finetuned_model.py
import os
from dotenv import load_dotenv

from src.data_loader import load_and_process_credit_data
from src.api_client import GeminiClient
from src.prompts import create_debiased_llm_prompt

def main():
    """
    Loads the fine-tuned model and tests it with a sample applicant.
    """
    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    if not project_id:
        print("🔥 Error: GOOGLE_CLOUD_PROJECT_ID not found in .env file.")
        return

    # --- IMPORTANT: Use the 'Tuned Model Name' from the finetuning script output ---
    # It should look like: projects/PROJECT_ID/locations/LOCATION/models/MODEL_ID
    TUNED_MODEL_NAME = "projects/1062357959737/locations/us-central1/endpoints/239116291250585600" 
    # Note: I have removed the "@1" version number, as the SDK handles this.

    if "your-tuned-model-name-here" in TUNED_MODEL_NAME:
        print("🚨 Please update the TUNED_MODEL_NAME variable in this script with your model name.")
        return

    # Initialize the client
    client = GeminiClient(project_id=project_id)
    
    # Load our specific fine-tuned model
    client.load_tuned_model(TUNED_MODEL_NAME)

    # Load the data to get a test case
    credit_df = load_and_process_credit_data()
    if credit_df is None:
        return

    # Select a sample customer to test
    test_customer = credit_df.iloc[20].to_dict()
    ground_truth = "Bad" if test_customer['Risk'] == 1 else "Good"
    
    print("\n--- 🧪 Testing Fine-Tuned Model ---")
    print(f"Ground Truth for applicant #21: {ground_truth}")

    # Use the same debiased prompt structure
    test_prompt = create_debiased_llm_prompt(test_customer)
    
    # Get the assessment from our fine-tuned model
    assessment = client.get_llm_assessment(test_prompt, use_tuned_model=True)
    
    print("\n--- Model Prediction ---")
    print(assessment)


if __name__ == "__main__":
    main()
