import pandas as pd
import numpy as np
import joblib 
import argparse
from random_forest import FeatureEngineering


class ModelTester:
    def __init__(self, test_csv, model='model.pkl', columns_file ='columns.pkl'):
        self.test_csv = test_csv
        self.model = model
        self.columns_file = columns_file
        
    def load_model(self):
        self.train_model = joblib.load(self.model)
        self.columns = joblib.load(self.columns_file)


    def read_csv(self):
        self.test = pd.read_csv(self.test_csv)

    def preprocess_data(self):
        self.test_ids = self.test['ID']
        feature_engineer = FeatureEngineering(self.test)
        self.test = feature_engineer.clean_data()
        self.test = self.test[self.columns]
        self.test_data = np.array(self.test)
        
            
    def make_predictions(self):
        test_predictions = self.train_model.predict(self.test_data)
        test_predictions_proba = self.train_model.predict_proba(self.test_data)[:, 1]
        return test_predictions, test_predictions_proba
        
        
        
    def save_predictions(self):
        test_predictions, test_predictions_proba = self.make_predictions()
        self.test['target'] = test_predictions
    
        my_result = pd.DataFrame({
            'ID': self.test_ids,
            'target': test_predictions 
            })
        my_result.to_csv('baseline_submission.csv', index=False)
        
        # class_counts = pd.Series(test_predictions).value_counts().sort_index()
        # print("Number of each class predicted:")
        # print(class_counts)


# if __name__ == "__main__":
#     tester = LogisticRegressionModelTester('test.csv')
#     tester.load_model()
#     tester.read_csv()
#     tester.preprocess_data()
#     tester.save_predictions()
#     print("Predictions made and saved")

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Test a loan default prediction model.')
        parser.add_argument('test_csv', type=str, help='Path to the test CSV file. Please ensure it has same columns . Check VariableDefinitions.txt file for more information')
        
    
        args = parser.parse_args()
        
        tester = ModelTester(args.test_csv)
        tester.load_model()
        tester.read_csv()
        tester.preprocess_data()
        tester.save_predictions()
        print("Model tested and prediction saved")