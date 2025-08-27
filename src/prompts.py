# src/prompts.py

def create_baseline_llm_prompt(customer_data: dict) -> str:
    """
    Creates a baseline prompt that asks the LLM to assess delinquency risk,
    including both financial and demographic (potentially biasing) information.
    """
    prompt = f"""
    Analyze the following credit applicant's profile to assess the risk of delinquency (late payment).
    Your response must follow this exact format:
    Verdict: [Good or Bad]
    Justification: [Your brief reasoning here, considering all provided data]

    Applicant Profile:
    - Age: {customer_data.get('age', 'N/A')}
    - Gender: {customer_data.get('gender', 'N/A')}
    - State of Residence: {customer_data.get('primary_state', 'N/A')}
    - Credit Limit (NGN): {customer_data.get('credit_limit_facility_amount_global_limit', 0):,.2f}
    - Average Credit Utilization: {customer_data.get('average_utilization', 0):.2%}

    Response:
    """
    return prompt

def create_debiased_llm_prompt(customer_data: dict) -> str:
    """
    Creates an advanced prompt engineered to mitigate bias by instructing the LLM
    to focus only on financial data for its delinquency risk assessment.
    """
    prompt = f"""
    As an expert financial risk analyst, provide an unbiased assessment of delinquency risk (Good or Bad).
    Your decision must be based ONLY on the applicant's financial data. Do not consider Age, Gender, or State.
    Your response must follow this exact format:
    Verdict: [Good or Bad]
    Justification: [Your brief reasoning based only on financial data]

    Financial Data:
    - Credit Limit (NGN): {customer_data.get('credit_limit_facility_amount_global_limit', 0):,.2f}
    - Average Credit Utilization: {customer_data.get('average_utilization', 0):.2%}

    Response:
    """
    return prompt
