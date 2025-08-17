# In your main UI file (e.g., app.py)
import streamlit as st

st.sidebar.header("Lender Risk Parameters")

# Create the input fields based on the screenshot 
max_outstanding = st.sidebar.number_input(
    "Maximum outstanding loan value", 
    min_value=0, 
    value=5000000
)
total_active_loans = st.sidebar.number_input(
    "Maximum total active loans", 
    min_value=0, 
    value=5
)
min_income = st.sidebar.number_input(
    "Minimum average monthly income", 
    min_value=0, 
    value=150000
)
credit_sources = st.sidebar.multiselect(
    "Choose preferred credit history source(s)",
    ['CRC', 'First Central', 'CreditRegistry'],
    default=['CRC', 'First Central']
)

# Store parameters in a dictionary
lender_parameters = {
    "max_outstanding": max_outstanding,
    "total_active_loans": total_active_loans,
    "min_income": min_income,
    "credit_sources": credit_sources
}