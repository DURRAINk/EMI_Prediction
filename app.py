import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

sclr = joblib.load('scaler.pkl')
emi_regressor_model = xgb.XGBRegressor()
emi_classifier_model = xgb.XGBClassifier()
# Load trained models
emi_regressor_model.load_model('emi_regressor_model.json')
emi_classifier_model.load_model('emi_classifier_model.json')

cat_mappings = {'marital_status':{'Married':1,'Single':0},
                'gender':{'m':1,'f':0},
                'education':{'Professional':3, 'Graduate':1, 'High School':0, 'Post Graduate':2},
                'employment_type': {'Private':1, 'Government':2, 'Self-employed':0},
                'company_type': {'Mid-size':2, 'MNC':4, 'Startup':0, 'Large Indian':3, 'Small':1},
                'house_type': {'Rented':0, 'Family':1, 'Own':2},
                'existing_loans': {'Yes':1, 'No':0},
                'emi_scenario': {'Personal Loan EMI':0, 'E-commerce Shopping EMI':4, 'Education EMI':2,
                                'Vehicle EMI':3, 'Home Appliances EMI':1},
                'emi_eligibility': {'Not_Eligible':0, 'Eligible':2, 'High_Risk':1}
                }
target_mapping = {0:'Not_Eligible', 1:'High_Risk', 2:'Eligible'}

def preprocess_input(dict_input):
    samp = pd.DataFrame(dict_input,index=[0])

    samp['emi_to_income'] = samp['current_emi_amount'] / samp['monthly_salary']
    samp['expense_to_income'] = samp['monthly_expenses'] / samp['monthly_salary']
    samp['emi_to_balance'] = samp['current_emi_amount'] / samp['bank_balance']

    # categorical mapping
    for col, mapping in cat_mappings.items():
        if col == 'emi_eligibility':
            continue
        samp[col] = samp[col].map(mapping)
   
    samp = sclr.transform(samp)
    return samp

## Application UI
st.set_page_config(page_title="EMI Prediction App", layout="centered")
st.title("EMI Prediction App")

st.write("This app predicts the EMI_Eligibility and Max_Monthly_EMI based on user inputs.")
st.write("Please enter the required details below:")


with st.form("Details Form"):
    col1, col2 = st.columns(2)

    # Column 1 inputs
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, step=1)
        gender = st.selectbox("Gender", cat_mappings["gender"].keys())
        marital_status = st.selectbox("Marital Status",cat_mappings["marital_status"].keys())
        education = st.selectbox("Education", cat_mappings["education"].keys())
        monthly_salary = st.number_input("Monthly Salary", min_value=0, step=1000)
        employment_type = st.selectbox("Employment Type",   cat_mappings["employment_type"].keys())
        years_of_employment = st.number_input("Years of Employment", min_value=0, max_value=50, step=1)
        company_type = st.selectbox("Company Type", cat_mappings["company_type"].keys())
        house_type = st.selectbox("House Type", cat_mappings["house_type"].keys())
        monthly_expenses = st.number_input("Monthly Expenses", min_value=0, step=500)

    # Column 2 inputs
    with col2:
        family_size = st.number_input("Family Size", min_value=1, max_value=20, step=1)
        existing_loans = st.selectbox("Number of Existing Loans",['No','Yes'])
        current_emi_amount = st.number_input("Current EMI Amount", min_value=0, step=500)
        credit_score = st.number_input("Credit Score", min_value=0, max_value=900, step=1)
        bank_balance = st.number_input("Bank Balance", min_value=0, step=1000)
        emergency_fund = st.number_input("Emergency Fund", min_value=0, step=1000)
        emi_scenario = st.selectbox("EMI Scenario", cat_mappings["emi_scenario"].keys())
        requested_amount = st.number_input("Requested Loan Amount", min_value=1000, step=1000)
        requested_tenure = st.number_input("Requested Tenure (months)", min_value=6, max_value=360, step=6)

    # Submit button
    submitted = st.form_submit_button("Predict")

if submitted:
    st.success("Details submitted successfully!")
    input_data = preprocess_input({
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "monthly_salary": monthly_salary,
        "employment_type": employment_type,
        "years_of_employment": years_of_employment,
        "company_type": company_type,
        "house_type": house_type,
        "family_size": family_size,
        "existing_loans": existing_loans,
        "current_emi_amount": current_emi_amount,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
        "emi_scenario": emi_scenario,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,
        "monthly_expenses": monthly_expenses
    })
    class_pred = emi_classifier_model.predict(input_data)
    emi_pred = emi_regressor_model.predict(input_data)

    cl1 , cl2 = st.columns(2)

    cl1.success(f"Predicted EMI_Eligibility: {target_mapping[class_pred[0]]}")

    cl2.success(f"Predicted Max_Monthly_EMI: {emi_pred[0]:.2f}")


