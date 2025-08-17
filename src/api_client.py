# src/api_client.py
import vertexai
from vertexai.generative_models import GenerativeModel, Content, Part

class GeminiClient:
    """
    Handles all communication with the Gemini API via the Vertex AI SDK.
    This client can manage both a base model and a fine-tuned model.
    """
    def __init__(self, project_id: str, location: str = "us-central1"):
        """
        Initializes the client and loads the default base model.
        """
        self.base_model = None
        self.tuned_model = None
        
        try:
            vertexai.init(project=project_id, location=location)
            self.base_model = GenerativeModel("gemini-2.5-flash")
            print("✅ Gemini Client initialized successfully with base model.")
        except Exception as e:
            print(f"🔥 Error initializing Gemini Client: {e}")

    def load_tuned_model(self, tuned_model_name: str):
        """
        Loads a fine-tuned model from Vertex AI using its full resource name.

        Args:
            tuned_model_name: The resource name of the fine-tuned model, 
                              e.g., "projects/PROJECT_ID/locations/LOCATION/models/MODEL_ID".
        """
        if not tuned_model_name:
            print("🚨 No tuned model name provided.")
            return

        try:
            # Correct Method: Load the fine-tuned model directly using GenerativeModel
            self.tuned_model = GenerativeModel(tuned_model_name)
            print(f"✅ Fine-tuned model loaded successfully: {tuned_model_name}")
        except Exception as e:
            print(f"🔥 Error loading fine-tuned model: {e}")

    def get_llm_assessment(self, prompt: str, use_tuned_model: bool = False) -> str:
        """
        Sends a prompt to the specified Gemini model and returns the text response.
        """
        model_to_use = self.tuned_model if use_tuned_model else self.base_model
        
        if not model_to_use:
            error_msg = "Fine-tuned model" if use_tuned_model else "Base model"
            return f"LLM Error: {error_msg} is not available."
            
        try:
            # The fine-tuned model expects a structured conversational format.
            # It's good practice to use this for the base model as well for consistency.
            request = [Content(role="user", parts=[Part.from_text(prompt)])]
            response = model_to_use.generate_content(request)
            return response.text
        except Exception as e:
            return f"An error occurred with the LLM API call: {e}"
