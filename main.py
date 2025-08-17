# main.py
import os
import time
import pandas as pd
from dotenv import load_dotenv

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
    TUNED_MODEL_ENDPOINT_NAME = "projects/1062357959737/locations/us-central1/endpoints/239116291250585600"

    # --- 1. Load Data & Train Benchmark Model ---
    credit_df = load_and_process_credit_data()
    if credit_df is None: return
    train_and_save_model(credit_df.copy())
    
    # --- 2. Initialize Clients and Load All Models ---
    ml_scorer = MLRiskScorer()
    llm_client = GeminiClient(project_id=project_id)
    llm_client.load_tuned_model(TUNED_MODEL_ENDPOINT_NAME)
    
    analysis_records = []
    sample_df = credit_df.head(1000)

    print("\n--- 🚀 Starting Rigorous Accuracy and Fairness Analysis (4 Models) ---")
    for index, customer in sample_df.iterrows():
        original_dict = customer.to_dict()
        counterfactual_dict = create_counterfactual(customer, bias_variable='Age').to_dict()
        print(f"\nProcessing applicant #{index+1}")

        ground_truth = 'Bad' if original_dict['Risk'] == 1 else 'Good'

        # --- Get predictions for both original and counterfactual profiles ---
        # 1. Benchmark ML Model
        ml_risk_orig = ml_scorer.predict_risk(original_dict)
        ml_risk_cf = ml_scorer.predict_risk(counterfactual_dict)

        # 2. Baseline LLM
        baseline_resp_orig = llm_client.get_llm_assessment(create_baseline_llm_prompt(original_dict), use_tuned_model=False)
        baseline_risk_orig = parse_llm_output(baseline_resp_orig)
        time.sleep(1.5)
        baseline_resp_cf = llm_client.get_llm_assessment(create_baseline_llm_prompt(counterfactual_dict), use_tuned_model=False)
        baseline_risk_cf = parse_llm_output(baseline_resp_cf)
        time.sleep(1.5)

        # 3. Debiased LLM (Prompt-Tuned)
        debiased_resp_orig = llm_client.get_llm_assessment(create_debiased_llm_prompt(original_dict), use_tuned_model=False)
        debiased_risk_orig = parse_llm_output(debiased_resp_orig)
        time.sleep(1.5)
        debiased_resp_cf = llm_client.get_llm_assessment(create_debiased_llm_prompt(counterfactual_dict), use_tuned_model=False)
        debiased_risk_cf = parse_llm_output(debiased_resp_cf)
        time.sleep(1.5)

        # 4. Fine-Tuned LLM (SFT Model)
        finetuned_resp_orig = llm_client.get_llm_assessment(create_debiased_llm_prompt(original_dict), use_tuned_model=True)
        finetuned_risk_orig = parse_llm_output(finetuned_resp_orig)
        time.sleep(1.5)
        finetuned_resp_cf = llm_client.get_llm_assessment(create_debiased_llm_prompt(counterfactual_dict), use_tuned_model=True)
        finetuned_risk_cf = parse_llm_output(finetuned_resp_cf)
        time.sleep(1.5)

        analysis_records.append({
            'ground_truth': ground_truth,
            'ml_pred': ml_risk_orig,
            'baseline_llm_pred': baseline_risk_orig,
            'debiased_llm_pred': debiased_risk_orig,
            'finetuned_llm_pred': finetuned_risk_orig,
            'ml_fairness_change': ml_risk_orig != ml_risk_cf,
            'baseline_llm_fairness_change': baseline_risk_orig != baseline_risk_cf,
            'debiased_llm_fairness_change': debiased_risk_orig != debiased_risk_cf,
            'finetuned_llm_fairness_change': finetuned_risk_orig != finetuned_risk_cf,
        })

    # --- 3. Calculate Final Metrics ---
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
            results_df['ml_fairness_change'].mean(),
            results_df['baseline_llm_fairness_change'].mean(),
            results_df['debiased_llm_fairness_change'].mean(),
            results_df['finetuned_llm_fairness_change'].mean()
        ]
    }
    metrics_df = pd.DataFrame(metrics)
    
    print("\n--- 📊 Final Performance Metrics ---")
    print(metrics_df)

    metrics_df.to_csv('final_metrics_results.csv', index=False)
    print("\n✅ Final metrics saved to final_metrics_results.csv")
    
    # --- 4. Visualize the Trade-off ---
    plot_final_analysis(metrics_df)

if __name__ == "__main__":
    main()
