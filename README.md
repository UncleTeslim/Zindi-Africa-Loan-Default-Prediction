# Loan Default Prediction 🎯

Predicting the likelihood of loan defaults using machine learning. This project is part of the Zindi Africa Challenge and aims to build robust predictive models to assist lenders in making informed decisions.

---

## 🖌️ Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Modeling Approach](#modeling-approach)
5. [Setup and Installation](#setup-and-installation)
6. [Usage](#usage)
7. [Results](#results)
9. [License](#license)

---

## 🌟 Overview

Loan default prediction is a critical application of machine learning in the financial sector. This project leverages historical loan data to build predictive models that estimate the likelihood of a borrower defaulting on a loan. 

By using this solution, financial institutions can reduce risks and improve decision-making in loan approvals.

---

## ✨ Features

- **Data Preprocessing:** Handling missing values, outliers, and feature engineering.
- **Exploratory Data Analysis (EDA):** Visualization and statistical analysis of data trends.
- **Model Development:** Implementation of machine learning algorithms for classification.
- **Evaluation Metrics:** Assess model performance using metrics such as AUC, F1 Score, and Precision-Recall.

---

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
4. **Model Evaluation:**  
   - Area Under Curve (AUC).  
   - Confusion Matrix.  
   - F1 Score for imbalanced data.  

---


## 📊 Results

| Metric          | Value      |
|------------------|------------|
| Accuracy         | 95.44%  |
| Precision        | 97% & 94%  |
| Recall           | 94% and 97% |
| F1 Score         |95-96%    |
| AUC-ROC          | 98.8%      |

Visualizations and detailed analysis can be found in the [new_zindi.ipynb] notebook.

Training script alone can be found in /final_submission/train.py

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

---
