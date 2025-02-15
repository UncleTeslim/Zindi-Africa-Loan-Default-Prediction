import pandas as pd
import joblib
from random_forest import FeatureEngineering

def load_model_and_columns(model='model.pkl', columns_file='columns.pkl'):
    model = joblib.load(model)
    columns = joblib.load(columns_file)
    return model, columns

def preprocess_input(input_data, columns):
    input_df = pd.DataFrame([input_data])
    feature_engineer = FeatureEngineering(input_df)
    input_df = feature_engineer.clean_data()
    input_df = input_df[columns]
    return input_df

def predict_loan_default(model, input_data):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]
    return prediction, prediction_proba

def collect_input():
    input_data = {}
    
    print("Please provide the following details:")
    input_data['ID'] = int(input("ID (A unique identifier for each entry): "))
    input_data['customer_id'] = float(input("Customer ID (Unique identifier for each customer): "))
    input_data['country_id'] = input("Country ID (Identifier or code representing the country): ")
    input_data['tbl_loan_id'] = float(input("Loan ID (Unique identifier for each loan): "))
    input_data['Total_Amount'] = float(input("Total Amount (The total loan amount initially disbursed): "))
    input_data['Total_Amount_to_Repay'] = float(input("Total Amount to Repay (The total amount to repay, including principal, interest, and fees): "))
    input_data['loan_type'] = input("Loan Type (e.g., Type_1, Type_20, etc.): ")
    input_data['disbursement_date'] = int(input("Disbursement Date (The date when the loan was disbursed, e.g., 20230101): "))
    input_data['duration'] = float(input("Duration (The length of the loan term in days): "))
    input_data['lender_id'] = float(input("Lender ID (Unique identifier for the lender): "))
    input_data['New_versus_Repeat'] = str(input("New versus Repeat (Repeat Loan or New Loan): "))
    input_data['Amount_Funded_By_Lender'] = float(input("Amount Funded by Lender (The portion of the loan funded directly by the lender): "))
    input_data['Lender_portion_Funded'] = float(input("Lender Portion Funded (Percentage of the total loan amount funded by the lender): "))
    input_data['due_date'] = int(input("Due Date (The date by which the loan repayment is due, e.g., 20240101): "))
    input_data['Lender_portion_to_be_repaid'] = float(input("Lender Portion to be Repaid (The portion of the outstanding loan to be repaid to the lender): "))
    
    return input_data

def main():
    model, columns = load_model_and_columns()
    input_data = collect_input()
    input_df = preprocess_input(input_data, columns)
    prediction, prediction_proba = predict_loan_default(model, input_df)
    
    if prediction[0] == 1:
        print("Prediction: The loan will default.")
    else:
        print("Prediction: The loan will not default.")
    
    #print(f"Confidence: {prediction_proba[0]:.4f}")

if __name__ == "__main__":
    main()