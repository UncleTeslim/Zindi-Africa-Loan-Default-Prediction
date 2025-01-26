# Loan Default Prediction 🎯

Predicting the likelihood of loan defaults using machine learning. This project is part of the Zindi Africa Challenge and aims to build robust predictive models to assist lenders in making informed decisions.

---

## 🖌️ Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Cloning the Repository](#cloning_the_repository)
4. [Running the Scripts](#running_the_scripts):
5. [Files Description](#files_description)
6. [Dataset](#dataset)
7. [Modeling Approach](#modeling-approach)
8. [Results](#results)
9. [Contributing](#Contributing)
10. [License](#license)

---

## 🌟 Overview

Loan default prediction is a critical application of machine learning in the financial sector. This project leverages historical loan data to build predictive models that estimate the likelihood of a borrower defaulting on a loan. By using this solution, financial institutions can reduce risks and improve decision-making in loan approvals.

This project predicts whether a customer will default on a loan using both logistic regression and random forest models. It consists of four main scripts:

1. log_reg.py: Trains the logistic regression model and saves it to a file.

2. random_forest.py: Trains the random forest model and saves to a file.

3. test.py: Tests the trained model on a dataset with targets to evaluate its performance.

4. inference.py: Takes user input via the command-line interface (CLI) to predict whether a customer will default on a loan.

---

## Prerequisites

Before running the scripts, ensure you have the following installed:

- Python 3.7 or higher
- Required Python libraries: `pandas`, `numpy`, `scikit-learn`, `imblearn`, `joblib`, `argparse`

You can install the required libraries using the following command:

```bash
pip install pandas numpy scikit-learn imbalanced-learn joblib argparse
```

## Cloning the repository

To get started, clone this repository to your local machine:

```bash
git clone https://github.com/UncleTeslim/Zindi-Africa-Loan-Default-Prediction.git
cd loan-default-prediction
```

## Running the scripts
Running the Scripts:

### 1.Training the Model

The log_reg.py and random_forest.py scripts train the logistic regression and random forest model respectively using a training dataset and saves the trained model to a file (model.pkl).

Usage:
```bash
python log_reg.py train.csv
```
**OR**
```bash
python random_forest.py train.csv
```

train.csv: Path to the training dataset in CSV format.

This will:

1. Preprocess the data.

2. Train the logistic regression model.

3. Save the trained model to model.pkl.

4. Save the column names to columns.pkl for inference.
   


### 2.Training the Model
The test.py script tests the trained model on a dataset with targets and evaluates its performance using metrics like accuracy, classification report, and ROC AUC score.

Usage:
```bash
python test.py test.csv
```
test.csv: Path to the test dataset in CSV format.

This will:

1. Load the trained model from model.pkl.

2. Preprocess the test data.

3. Evaluate the model and print performance metrics.


### 3. Making predictions
The inference.py script takes user input via the CLI to predict whether a customer will default on a loan..

Usage:
```bash
python inference.py
```

This will prompt you to enter the following features:

Total_Amount: Total loan amount.

duration: Loan duration.

New_versus_Repeat: Whether the loan is new or repeat .

loan_type: Type of loan (encoded as an integer).

repayment_ratio: Ratio of total amount to repay to the total loan amount.

lender_funded_ratio: Ratio of the amount funded by the lender to the total loan amount.

lender_repaid_ratio: Ratio of the lender portion to be repaid to the lender portion funded.

The script will output the prediction (0 for no default, 1 for default).

***Please check VariableDefinitions.txt for inputs expected***


## File Descriptions:
train.py: Trains the logistic regression model and saves it to model.pkl.

test.py: Tests the trained model on a dataset with targets.

inference.py: Takes user input to make predictions using the trained model.

Variable Definitions.txt: Description of variables in Train/Test.csv

README.md: This documentation file.



## 🗂 Dataset

The dataset for this project contains the following:
- **Features:** Loan amount, disbursement date, amoount repaid, New or returning borrower, duration etc.
- **Target Variable:** Binary (1 = Default, 0 = No Default).

> **Note:** The dataset was provided by [Zindi Africa](https://zindi.africa/](https://zindi.africa/competitions/african-credit-scoring-challenge) as part of the Loan Default Prediction Challenge. Access the dataset directly from the Zindi platform.

---

## 🛠️ Modeling Approach

1. **Data Cleaning:**  
   - Handled missing values using imputation.
   - Dealt with outliers to reduce model bias.
2. **Feature Engineering:**  
   - Created meaningful features (e.g., repayment ratio).
   - One-hot encoding for categorical variables.
3. **Machine Learning Algorithms:**  
   - Logistic Regression
   - Random Forest Classifier 
4. **Model Evaluation:**  
   - Area Under Curve (AUC).  
   - Confusion Matrix.  
   - F1 Score for imbalanced data.  

---


## 📊 Results for Logistic Regression

| Metric          | Value      |
|------------------|------------|
| Accuracy         | 95.44%  |
| Precision        | 97% & 94%  |
| Recall           | 94% and 97% |
| F1 Score         |95-96%    |
| AUC-ROC          | 98.8%      |

Confusion Matrix:
[[18915  1304]\
 [  541 19678]]


## 📊 Results for Random Forest

| Metric          | Value      |
|------------------|------------|
| Accuracy         | 99.45%  |
| Precision        | 100% & 99%  |
| Recall           | 99% and 100% |
| F1 Score         |99% |
| AUC-ROC          | 99.9%      |

Confusion Matrix:
[[20072   147]\
 [   73 20146]]


Visualizations and detailed analysis can be found in the [new_zindi.ipynb] notebook.

---

## 🤝 Contributing

Contributions are welcome!  
1. Fork the repository.  
2. Create a new branch (`git checkout -b feature/your-feature`).  
3. Commit your changes (`git commit -m "Add your feature"`).  
4. Push to the branch (`git push origin feature/your-feature`).  
5. Open a Pull Request.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

```

---

### How to Use This Documentation

1. Save the above content in a file named `README.md` in the root of your GitHub repository.
2. Replace placeholders like `your-username` and `loan-default-prediction` with your actual GitHub username and repository name.
3. Push the changes to your repository.

This `README.md` will serve as the main documentation for your project, making it easy for others to understand and use your code.

---
```
