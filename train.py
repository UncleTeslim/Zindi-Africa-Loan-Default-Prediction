import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,  roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib 
import argparse



#OOP


class FeatureEngineering:
    def __init__(self, df):
        self.df = df
        
        
    def clean_data(self):
        self.df.drop(['ID'], axis=1, inplace=True)
        
        self.df['repayment_ratio'] = self.df['Total_Amount_to_Repay'] / self.df['Total_Amount']
        self.df['lender_funded_ratio'] = self.df['Amount_Funded_By_Lender'] / self.df['Total_Amount']
        self.df['lender_repaid_ratio'] = self.df['Lender_portion_to_be_repaid'] / self.df['Lender_portion_Funded']
        
        
        self.df = self.df.drop(['country_id', 'customer_id', 'tbl_loan_id', 'lender_id', 'disbursement_date', 'due_date',
                   'Amount_Funded_By_Lender', 'Lender_portion_Funded', 'Lender_portion_to_be_repaid',
                   'Total_Amount_to_Repay'], axis=1)
        
        
        self.df['lender_repaid_ratio'] = self.df['lender_repaid_ratio'].fillna(self.df['lender_repaid_ratio'].mean())
        self.df['lender_funded_ratio'] = self.df['lender_funded_ratio'].fillna(self.df['lender_funded_ratio'].mean())

        numeric_cols = ['Total_Amount', 'duration', 'lender_repaid_ratio', 'lender_funded_ratio']
        for col in numeric_cols:
            self.df[f'{col}_transformed'] = np.log1p(self.df[col])
            
        self.df = self.df.drop(['Total_Amount', 'duration', 'lender_repaid_ratio', 'lender_funded_ratio'], axis=1)
        
        self.df['New_versus_Repeat'] = self.df['New_versus_Repeat'].apply(lambda x: 1 if x == 'Repeat Loan' else 0)
        
        
        encoder = LabelEncoder()

        self.df['loan_type'] = encoder.fit_transform(self.df['loan_type'])
    
        
        return self.df




class LogisticRegressionModel:
    def __init__(self, train_csv, max_iter=1000, random_state=42, test_size=0.3, target='target'):
        self.max_iter = max_iter
        self.random_state = random_state
        self.test_size = test_size
        self.train_csv = train_csv
        self.target = target
        self.log_reg = LogisticRegression(max_iter=self.max_iter, random_state=self.random_state)
        
    
    
    def read_csv(self):
        self.train = pd.read_csv(self.train_csv)
        
        
    def preprocess_data(self):
        feature_engineer = FeatureEngineering(self.train)
        self.train = feature_engineer.clean_data()
        
        X = self.train.drop(self.target, axis=1)
        y = self.train[self.target]
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_resampled, y_resampled, test_size=self.test_size, 
         random_state=self.random_state, stratify=y_resampled)
        
        joblib.dump(X.columns, 'columns.pkl')
        
    def train_model(self):
        self.log_reg.fit(self.X_train, self.y_train)
        
        
    def evaluate_model(self):
        y_pred = self.log_reg.predict(self.X_test)
        y_probs = self.log_reg.predict_proba(self.X_test)[:, 1] 
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy:", accuracy)
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        roc_auc = roc_auc_score(self.y_test, y_probs)
        print(f"\nROC AUC Score: {roc_auc:.4f}")
    
    def save_model(self):
        joblib.dump(self.log_reg, 'logistic_regression_model.pkl')
        
    
# if __name__ == "__main__":
#     model = LogisticRegressionModel('train.csv')
#     model.read_csv()
#     model.preprocess_data()
#     model.train_model()
#     model.evaluate_model()
#     model.save_model()
#     print("Model trained and saved")
    
    
if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Train a loan default prediction model.')
        parser.add_argument('train_csv', type=str, help='Path to the training CSV file. Please ensure it has same columns . Check VariableDefinitions.txt file for more information')
        
        
        args = parser.parse_args()
        
        model = LogisticRegressionModel(args.train_csv)
        model.read_csv()
        model.preprocess_data()
        model.train_model()
        model.evaluate_model()
        model.save_model()
        print("Model trained and saved")