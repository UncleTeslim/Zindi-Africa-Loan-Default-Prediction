import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,  roc_auc_score
from imblearn.over_sampling import SMOTE


#Reading train and test dataset which i cleaned ,analysed 
# and encoded in jupyter notebook

train = pd.read_csv('clean_train.csv')

test = pd.read_csv('clean_test.csv')

#I declared the training features and targets. 
# I then applied SMOTE to balance the dataset.

X = train.drop('target', axis=1)
y = train['target']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


# Splitting the training data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)


log_reg = LogisticRegression(max_iter=1000, random_state=42 )

# Train the model
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)
y_probs = log_reg.predict_proba(X_test)[:, 1] 

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_probs)
print(f"\nROC AUC Score: {roc_auc:.4f}")
#print(test_predictions_proba[:10]) 


#Make predictiosns based on test dataset using my model

test_features = test[X_train.columns]

test_predictions = log_reg.predict(test_features)
test_predictions_proba = log_reg.predict_proba(test_features)[:, 1]



# #saving prediction to test dataset and extracting just ID and target

# test['target'] = test_predictions
# my_result =  test[['ID', 'target']]
# my_result.head(10)


# my_result.to_csv('baseline_submission.csv', index=False)
