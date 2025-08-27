# main.py
import os
import time
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Import all necessary modules from the src package
from src.data_loader import load_and_process_credit_data
from src.model_trainer import train_and_save_model
from src.risk_scorer import MLRiskScorer
from src.api_client import GeminiClient
from src.prompts import create_baseline_llm_prompt, create_debiased_llm_prompt
from src.bias_analyzer import create_counterfactual
from src.visualization import plot_final_analysis

def parse_llm_output(llm_response: str) -> str:
    """A robust function to extract 'Good' or 'Bad' from the LLM's text output."""
    response_lower = llm_response.strip().lower()
    if 'verdict:' in response_lower:
        verdict = response_lower.split('verdict:')[1].strip()
        if 'bad' in verdict: return 'Bad'
        if 'good' in verdict: return 'Good'
    # Fallback for less structured responses
    if 'bad' in response_lower: return 'Bad'
    if 'good' in response_lower: return 'Good'
    return 'Unknown'

def main():
    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    if not project_id:
        print("🔥 Error: GOOGLE_CLOUD_PROJECT_ID not found in .env file.")
        return

    # --- IMPORTANT: Paste your Tuned Model Endpoint Name here ---
    TUNED_MODEL_ENDPOINT_NAME = "projects/1062357959737/locations/us-central1/endpoints/5640515541211807744"

    # --- 1. Load Data & Split for Proper Evaluation ---
    customer_df = load_and_process_credit_data()
    if customer_df is None: return
    
    # Split data to prevent data leakage for the benchmark model
    train_df, test_df = train_test_split(customer_df, test_size=0.3, random_state=42, stratify=customer_df['isdelinquent'])
    
    # --- 2. Train the Benchmark ML Model on the Training Set ONLY ---
    train_and_save_model(train_df.copy())
    
    # --- 3. Initialize Clients and Load All Models ---
    ml_scorer = MLRiskScorer()
    llm_client = GeminiClient(project_id=project_id)
    llm_client.load_tuned_model(TUNED_MODEL_ENDPOINT_NAME)
    
    analysis_records = []
    # The experiment will now run on the unseen test data
    sample_df = test_df.head(1000)

    print("\n--- 🚀 Starting Comprehensive Accuracy and Fairness Analysis ---")
    for index, customer in sample_df.iterrows():
        original_dict = customer.to_dict()
        
        # Create counterfactuals for all sensitive attributes
        cf_age_dict = create_counterfactual(customer, bias_variable='age').to_dict()
        cf_gender_dict = create_counterfactual(customer, bias_variable='gender').to_dict()
        cf_state_dict = create_counterfactual(customer, bias_variable='primary_state').to_dict()
        
        print(f"\nProcessing applicant #{customer['customerid']}")

        ground_truth = 'Bad' if original_dict['isdelinquent'] == 1 else 'Good'

        # --- Get predictions for the original profile from all models ---
        ml_risk_orig = ml_scorer.predict_risk(original_dict)
        baseline_risk_orig = parse_llm_output(llm_client.get_llm_assessment(create_baseline_llm_prompt(original_dict)))
        time.sleep(2)
        debiased_risk_orig = parse_llm_output(llm_client.get_llm_assessment(create_debiased_llm_prompt(original_dict)))
        time.sleep(2)
        
        # --- DEBUGGING THE FINE-TUNED MODEL ---
        raw_response_finetuned = llm_client.get_llm_assessment(create_debiased_llm_prompt(original_dict), use_tuned_model=True)
        print(f"RAW RESPONSE FROM TUNED MODEL: ---> {raw_response_finetuned} <---") # This is the crucial line
        finetuned_risk_orig = parse_llm_output(raw_response_finetuned)
        time.sleep(2)
        # --- END DEBUGGING BLOCK ---

        # --- Get predictions for ALL counterfactuals to test fairness across ALL models ---
        # Age Counterfactuals
        ml_risk_cf_age = ml_scorer.predict_risk(cf_age_dict)
        baseline_risk_cf_age = parse_llm_output(llm_client.get_llm_assessment(create_baseline_llm_prompt(cf_age_dict)))
        time.sleep(2)
        debiased_risk_cf_age = parse_llm_output(llm_client.get_llm_assessment(create_debiased_llm_prompt(cf_age_dict)))
        time.sleep(2)
        finetuned_risk_cf_age = parse_llm_output(llm_client.get_llm_assessment(create_debiased_llm_prompt(cf_age_dict), use_tuned_model=True))
        time.sleep(2)

        # Gender Counterfactuals
        ml_risk_cf_gender = ml_scorer.predict_risk(cf_gender_dict)
        baseline_risk_cf_gender = parse_llm_output(llm_client.get_llm_assessment(create_baseline_llm_prompt(cf_gender_dict)))
        time.sleep(2)
        debiased_risk_cf_gender = parse_llm_output(llm_client.get_llm_assessment(create_debiased_llm_prompt(cf_gender_dict)))
        time.sleep(2)
        finetuned_risk_cf_gender = parse_llm_output(llm_client.get_llm_assessment(create_debiased_llm_prompt(cf_gender_dict), use_tuned_model=True))
        time.sleep(2)
        
        # State Counterfactuals
        ml_risk_cf_state = ml_scorer.predict_risk(cf_state_dict)
        baseline_risk_cf_state = parse_llm_output(llm_client.get_llm_assessment(create_baseline_llm_prompt(cf_state_dict)))
        time.sleep(2)
        debiased_risk_cf_state = parse_llm_output(llm_client.get_llm_assessment(create_debiased_llm_prompt(cf_state_dict)))
        time.sleep(2)
        finetuned_risk_cf_state = parse_llm_output(llm_client.get_llm_assessment(create_debiased_llm_prompt(cf_state_dict), use_tuned_model=True))
        time.sleep(2)
        
        analysis_records.append({
            'ground_truth': ground_truth,
            'ml_pred': ml_risk_orig,
            'baseline_llm_pred': baseline_risk_orig,
            'debiased_llm_pred': debiased_risk_orig,
            'finetuned_llm_pred': finetuned_risk_orig,
            'ml_fairness_change_age': ml_risk_orig != ml_risk_cf_age,
            'baseline_llm_fairness_change_age': baseline_risk_orig != baseline_risk_cf_age,
            'debiased_llm_fairness_change_age': debiased_risk_orig != debiased_risk_cf_age,
            'finetuned_llm_fairness_change_age': finetuned_risk_orig != finetuned_risk_cf_age,
            'ml_fairness_change_gender': ml_risk_orig != ml_risk_cf_gender,
            'baseline_llm_fairness_change_gender': baseline_risk_orig != baseline_risk_cf_gender,
            'debiased_llm_fairness_change_gender': debiased_risk_orig != debiased_risk_cf_gender,
            'finetuned_llm_fairness_change_gender': finetuned_risk_orig != finetuned_risk_cf_gender,
            'ml_fairness_change_state': ml_risk_orig != ml_risk_cf_state,
            'baseline_llm_fairness_change_state': baseline_risk_orig != baseline_risk_cf_state,
            'debiased_llm_fairness_change_state': debiased_risk_orig != debiased_risk_cf_state,
            'finetuned_llm_fairness_change_state': finetuned_risk_orig != finetuned_risk_cf_state,
        })

    # --- 4. Calculate Final Metrics ---
    results_df = pd.DataFrame(analysis_records)
    
    metrics = {
        'Model': ['Benchmark ML', 'Baseline LLM', 'Debiased LLM', 'Fine-Tuned LLM'],
        'Accuracy': [
            (results_df['ground_truth'] == results_df['ml_pred']).mean(),
            (results_df['ground_truth'] == results_df['baseline_llm_pred']).mean(),
            (results_df['ground_truth'] == results_df['debiased_llm_pred']).mean(),
            (results_df['ground_truth'] == results_df['finetuned_llm_pred']).mean()
        ],
        'Fairness Disparity (Age)': [
            results_df['ml_fairness_change_age'].mean(),
            results_df['baseline_llm_fairness_change_age'].mean(),
            results_df['debiased_llm_fairness_change_age'].mean(),
            results_df['finetuned_llm_fairness_change_age'].mean()
        ],
        'Fairness Disparity (Gender)': [
            results_df['ml_fairness_change_gender'].mean(),
            results_df['baseline_llm_fairness_change_gender'].mean(),
            results_df['debiased_llm_fairness_change_gender'].mean(),
            results_df['finetuned_llm_fairness_change_gender'].mean()
        ],
        'Fairness Disparity (State)': [
            results_df['ml_fairness_change_state'].mean(),
            results_df['baseline_llm_fairness_change_state'].mean(),
            results_df['debiased_llm_fairness_change_state'].mean(),
            results_df['finetuned_llm_fairness_change_state'].mean()
        ]
    }
    metrics_df = pd.DataFrame(metrics)
    
    print("\n--- 📊 Final Performance Metrics ---")
    print(metrics_df)

    metrics_df.to_csv('final_metrics_results.csv', index=False)
    print("\n✅ Final metrics saved to final_metrics_results.csv")
    
    # --- 5. Visualize the Trade-off ---
    plot_final_analysis(metrics_df)

if __name__ == "__main__":
    main()
