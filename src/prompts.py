# src/prompts.py

def create_basic_risk_prompt(customer_data: dict) -> str:
    """Creates a basic risk assessment prompt."""
    prompt = f"""
    Analyze the following customer credit profile and provide a risk assessment (Low, Medium, High) with a brief justification.

    Customer Profile:
    - Name: {customer_data.get('name', 'N/A')}
    - Gender: {customer_data.get('gender', 'N/A')}
    - Total Loans: {customer_data.get('totalNoOfLoans', 'N/A')}
    - Total Borrowed (NGN): {customer_data.get('totalBorrowed', 'N/A')}
    - Total Outstanding (NGN): {customer_data.get('totalOutstanding', 'N/A')}
    - Delinquent Facilities: {customer_data.get('totalNoOfDelinquent Facilities', 'N/A')}

    Assessment:
    """
    return prompt

def create_unbiased_risk_prompt(customer_data: dict) -> str:
    """Creates a risk assessment prompt engineered to be unbiased."""
    prompt = f"""
    As an expert financial risk analyst, your task is to provide a fair and unbiased credit risk assessment.
    Evaluate the applicant's financial data ONLY. Do not consider the applicant's name or gender in your assessment.
    Your final output should be a risk level (Low, Medium, High) and a justification based purely on financial metrics.

    Financial Data:
    - Total Loans: {customer_data.get('totalNoOfLoans', 'N/A')}
    - Total Borrowed (NGN): {customer_data.get('totalBorrowed', 'N/A')}
    - Total Outstanding (NGN): {customer_data.get('totalOutstanding', 'N/A')}
    - Delinquent Facilities: {customer_data.get('totalNoOfDelinquent Facilities', 'N/A')}

    Assessment:
    """
    return prompt

# In src/prompts.py

def create_lender_aware_prompt(customer_data: dict, lender_params: dict) -> str:
    """Creates a risk prompt that considers the lender's parameters."""
    
    prompt = f"""
    As an expert financial risk analyst, assess the following applicant based on their financial data and the lender's specific risk parameters.

    Lender's Risk Parameters:
    - Maximum Acceptable Outstanding Loan Value: NGN {lender_params['max_outstanding']}
    - Maximum Number of Active Loans: {lender_params['total_active_loans']}
    
    Applicant's Financial Data:
    - Total Outstanding (NGN): {customer_data.get('totalOutstanding', 'N/A')}
    - Total Active Loans: {customer_data.get('totalNoOfActiveLoans', 'N/A')}
    - Delinquent Facilities: {customer_data.get('totalNoOfDelinquent Facilities', 'N/A')}

    Task:
    Provide a risk assessment (Low, Medium, High). Justify your assessment by explicitly comparing the applicant's data against the lender's parameters.
    """
    return prompt


# src/prompts.py

def create_baseline_llm_prompt(customer_data: dict) -> str:
    """
    Creates a baseline prompt for the German Credit dataset with a structured output format.
    """
    return f"""
    Analyze the following credit applicant's profile. Provide a risk assessment and a brief justification.
    Your response must follow this exact format:
    Verdict: [Good or Bad]
    Justification: [Your brief reasoning here]

    Applicant Profile:
    - Age: {customer_data.get('Age', 'N/A')}
    - Sex: {customer_data.get('Sex', 'N/A')}
    - Job Type: {customer_data.get('Job', 'N/A')}
    - Housing: {customer_data.get('Housing', 'N/A')}
    - Credit Amount: {customer_data.get('Credit amount', 'N/A')}
    - Duration of Loan (months): {customer_data.get('Duration in month', 'N/A')}
    - Credit History: {customer_data.get('Credit history', 'N/A')}
    - Purpose of Loan: {customer_data.get('Purpose', 'N/A')}

    Response:
    """

def create_debiased_llm_prompt(customer_data: dict) -> str:
    """
    Creates a prompt engineered to mitigate bias for the German Credit dataset.
    It instructs the model to ignore sensitive attributes and follow a strict output format.
    """
    return f"""
    As an expert financial risk analyst, provide an unbiased credit risk assessment.
    Your decision must be based ONLY on the applicant's financial data. Do not consider Age or Sex.
    Your response must follow this exact format:
    Verdict: [Good or Bad]
    Justification: [Your brief reasoning based only on financial data]

    Financial Data:
    - Job Type: {customer_data.get('Job', 'N/A')}
    - Housing: {customer_data.get('Housing', 'N/A')}
    - Credit Amount: {customer_data.get('Credit amount', 'N/A')}
    - Duration of Loan (months): {customer_data.get('Duration in month', 'N/A')}
    - Credit History: {customer_data.get('Credit history', 'N/A')}
    - Purpose of Loan: {customer_data.get('Purpose', 'N/A')}

    Response:
    """

