# run_finetuning_job.py
import os
import time
from dotenv import load_dotenv
import vertexai
from vertexai.tuning import sft

def start_finetuning_job(project_id: str, location: str, train_gcs_uri: str, validation_gcs_uri: str):
    """
    Starts a supervised fine-tuning job on Vertex AI using a training and validation dataset.
    
    Args:
        project_id: Your Google Cloud Project ID.
        location: The GCP region for the job (e.g., "us-central1").
        train_gcs_uri: The GCS URI of your training JSONL dataset.
        validation_gcs_uri: The GCS URI of your validation JSONL dataset.
    """
    print(f"Starting fine-tuning job in project '{project_id}'...")
    
    vertexai.init(project=project_id, location=location)

    try:
        # Start the tuning job using the sft.train method
        sft_tuning_job = sft.train(
            source_model="gemini-2.5-flash",
            train_dataset=train_gcs_uri,
            # --- Overfitting Mitigation: Use a validation dataset ---
            validation_dataset=validation_gcs_uri,
            # You can also adjust epochs (e.g., epochs=3) for more control
            epochs=4 
        )

        print("✅ Fine-tuning job submitted successfully! Now polling for completion...")
        print(f"You can also monitor the job in the Google Cloud Console to see training vs. validation loss.")

        # Polling for job completion
        while not sft_tuning_job.has_ended:
            print("   - Job has not ended yet. Waiting for 60 seconds...")
            time.sleep(60)
            sft_tuning_job.refresh()

        print("\n🎉 Fine-tuning job completed!")
        print(f"Tuned Model Name: {sft_tuning_job.tuned_model_name}")
        print(f"Tuned Model Endpoint Name: {sft_tuning_job.tuned_model_endpoint_name}")
        print(f"Experiment Details: {sft_tuning_job.experiment}")


    except Exception as e:
        print(f"🔥 An error occurred while submitting the fine-tuning job: {e}")


if __name__ == "__main__":
    load_dotenv()
    gcp_project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")

    if not gcp_project_id:
        print("🔥 Error: GOOGLE_CLOUD_PROJECT_ID not found in .env file.")
    else:
        # --- IMPORTANT ---
        # You must first upload both 'finetuning_dataset_train.jsonl' and 
        # 'finetuning_dataset_validation.jsonl' to a GCS bucket.
        # Then, replace the URIs below with the correct paths.
        gcs_train_uri = "gs://gemini-name-credit-risk-bucket-2025/finetuning_dataset_train.jsonl"
        gcs_validation_uri = "gs://gemini-name-credit-risk-bucket-2025/finetuning_dataset_validation.jsonl"
        
        if "your-bucket-name" in gcs_train_uri:
            print("🚨 Please update the GCS bucket URIs in 'run_finetuning_job.py' before running.")
        else:
            start_finetuning_job(
                project_id=gcp_project_id,
                location="us-central1",
                train_gcs_uri=gcs_train_uri,
                validation_gcs_uri=gcs_validation_uri
            )
