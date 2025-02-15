import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib 
import argparse


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
        self.df = self.df.drop(['Total_Amount', 'duration', 'lender_repaid_ratio', 'lender_funded_ratio', 'New_versus_Repeat'], axis=1)
        encoder = LabelEncoder()
        self.df['loan_type'] = encoder.fit_transform(self.df['loan_type'])
        return self.df


class RandomForest:
    def __init__(self, train_csv, n_estimators=100, random_state=42, max_depth=None, test_size=0.3, target='target', Total_Amount_transformed='Total_Amount_transformed'):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.test_size = test_size
        self.train_csv = train_csv
        self.target = target
        self.Total_Amount_transformed = Total_Amount_transformed
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators, 
            random_state=self.random_state, 
            max_depth=self.max_depth
            )

    def read_csv(self):
        self.train = pd.read_csv(self.train_csv)

    def preprocess_data(self):
        feature_engineer = FeatureEngineering(self.train)
        self.train = feature_engineer.clean_data()
        X = self.train.drop([self.target, self.Total_Amount_transformed], axis=1)
        y = self.train[self.target]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)
        
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        joblib.dump(X.columns, 'columns.pkl')
    
    

    def train_model(self):
        self.rf.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.rf.predict(self.X_test)
        y_probs = self.rf.predict_proba(self.X_test)[:, 1]
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy:", accuracy)
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        roc_auc = roc_auc_score(self.y_test, y_probs)
        print(f"\nROC AUC Score: {roc_auc:.4f}")

    def save_model(self):
        joblib.dump(self.rf, 'model.pkl')

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Train a loan default prediction model.')
        parser.add_argument('train_csv', type=str, help='Path to the training CSV file.')
        args = parser.parse_args()
        model = RandomForest(args.train_csv)
        model.read_csv()
        model.preprocess_data()
        model.train_model()
        model.evaluate_model()
        model.save_model()
        print("Model trained and saved")
