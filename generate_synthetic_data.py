# generate_synthetic_data.py
import pandas as pd
import numpy as np
import json
import random
from faker import Faker

# Initialize Faker for generating random data
fake = Faker()

# --- Configuration for Realistic Nigerian Data ---
# Expanded to include all 36 states and the FCT for better geographic diversity
NIGERIAN_STATES = [
    "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue", "Borno",
    "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu", "Gombe", "Imo",
    "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi", "Kwara", "Lagos",
    "Nasarawa", "Niger", "Ogun", "Ondo", "Osun", "Oyo", "Plateau", "Rivers",
    "Sokoto", "Taraba", "Yobe", "Zamfara", "Abuja FCT"
]
PHONE_PREFIXES = ["0803", "0805", "0807", "0703", "0706", "0810", "0813", "0909", "0903", "0802"]

# Expanded lists to include more names from major ethnic groups (Yoruba, Igbo, Hausa, etc.)
MALE_NAMES = [
    "Adekunle", "Emeka", "Musa", "Oluwaseun", "Chinedu", "Ibrahim", "Tunde", "Abubakar",
    "Chukwuma", "Femi", "Garba", "Jide", "Nnamdi", "Rotimi", "Sani", "Tariq", "Yakubu",
    "Obinna", "Dayo", "Haruna", "Ikenna", "Babatunde", "Usman", "Gozie", "Segun"
]
FEMALE_NAMES = [
    "Aisha", "Funke", "Ngozi", "Fatima", "Chioma", "Bolanle", "Hadiza", "Adanna",
    "Bukola", "Chiamaka", "Halima", "Ifunanya", "Kemi", "Nneka", "Omolara", "Zainab",
    "Amina", "Bisola", "Ezinne", "Folake", "Habiba", "Nkechi", "Simisola", "Yetunde"
]
SURNAMES = [
    "Adeyemi", "Okafor", "Bello", "Okoro", "Abubakar", "Popoola", "Eze", "Adewale",
    "Nwosu", "Lawal", "Onuoha", "Balogun", "Igwe", "Mohammed", "Ogunleye", "Okeke",
    "Salami", "Umar", "Akinwumi", "Chukwu", "Danjuma", "Obi", "Shittu", "Uche"
]

def generate_bvn():
    """Generates a random, realistic-looking 11-digit BVN."""
    return str(random.randint(10**10, (10**11) - 1))

def generate_financial_json(base_value, num_sources=3):
    """Generates the complex JSON string for financial columns."""
    sources = ["CRC", "FIRST_CENTRAL", "CREDIT_REGISTRY"]
    data = []
    for i in range(num_sources):
        value = int(base_value * random.uniform(0.8, 1.2))
        data.append({"source": sources[i], "value": value})
    return json.dumps(data).replace('"', '""')

def generate_loan_performance(is_delinquent):
    """Generates realistic loan performance data."""
    num_loans = random.randint(1, 5)
    loans = []
    for _ in range(num_loans):
        if is_delinquent and len(loans) == num_loans - 1:
            status = "Non-Performing"
        else:
            status = random.choice(["Performing", "Performing", "Performing", "Non-Performing"])
        
        loans.append({
            "loanProvider": fake.company(),
            "loanAmount": random.randint(50000, 5000000),
            "performanceStatus": status
        })
    
    data = [{"source": "CRC", "value": loans}]
    return json.dumps(data).replace('"', '""')

def generate_synthetic_dataset(num_records=1000):
    """Generates a full synthetic dataset with the specified number of records."""
    print(f"Generating {num_records} synthetic records...")
    data = []
    for _ in range(num_records):
        gender = random.choice(["Male", "Female"])
        name = f"{random.choice(MALE_NAMES if gender == 'Male' else FEMALE_NAMES)} {random.choice(SURNAMES)}"
        
        is_delinquent = random.random() < 0.2
        total_overdue = random.randint(10000, 500000) if is_delinquent else 0
        
        record = {
            'bvn': generate_bvn(), # Use the custom function here
            'name': name,
            'gender': gender,
            'dateOfBirth': fake.date_of_birth(minimum_age=20, maximum_age=70).strftime('%d/%m/%Y'),
            'address': f"{fake.street_address()}, {random.choice(NIGERIAN_STATES)}",
            'phone': f"{random.choice(PHONE_PREFIXES)}{fake.msisdn()[4:]}",
            'email': fake.email(),
            'totalNoOfLoans': generate_financial_json(random.randint(2, 25)),
            'totalNoOfInstitutions': generate_financial_json(random.randint(1, 10)),
            'totalNoOfActiveLoans': generate_financial_json(random.randint(0, 10)),
            'totalNoOfClosedLoans': generate_financial_json(random.randint(1, 15)),
            'totalBorrowed': generate_financial_json(random.randint(100000, 20000000)),
            'totalOutstanding': generate_financial_json(random.randint(0, 10000000)),
            'totalOverdue': generate_financial_json(total_overdue),
            'loanPerformance': generate_loan_performance(is_delinquent)
        }
        data.append(record)
    
    df = pd.DataFrame(data)
    output_path = 'data/Nigerian_Credit_Data_1000.csv'
    df.to_csv(output_path, index=False)
    print(f"✅ Synthetic data successfully saved to {output_path}")

if __name__ == "__main__":
    generate_synthetic_dataset()
