from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
from random_forest import FeatureEngineering

def load_model_and_columns(model='model.pkl', columns_file='columns.pkl'):
    model = joblib.load(model)
    columns = joblib.load(columns_file)
    return model, columns

model, columns = load_model_and_columns()


app = Flask(__name__)


def preprocess_input(input_data, columns):
    input_df = pd.DataFrame([input_data])
    feature_engineer = FeatureEngineering(input_df)
    input_df = feature_engineer.clean_data()
    input_df = input_df[columns]
    return input_df


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            input_data = {
                "ID": int(request.form["ID"]),
                "customer_id": float(request.form["customer_id"]),
                "country_id": request.form["country_id"],
                "tbl_loan_id": float(request.form["tbl_loan_id"]),
                "Total_Amount": float(request.form["Total_Amount"]),
                "Total_Amount_to_Repay": float(request.form["Total_Amount_to_Repay"]),
                "loan_type": request.form["loan_type"],
                "disbursement_date": int(request.form["disbursement_date"]),
                "duration": float(request.form["duration"]),
                "lender_id": float(request.form["lender_id"]),
                "New_versus_Repeat": request.form["New_versus_Repeat"],
                "Amount_Funded_By_Lender": float(request.form["Amount_Funded_By_Lender"]),
                "Lender_portion_Funded": float(request.form["Lender_portion_Funded"]),
                "due_date": int(request.form["due_date"]),
                "Lender_portion_to_be_repaid": float(request.form["Lender_portion_to_be_repaid"])
            }

            # Preprocess input
            input_df = preprocess_input(input_data, columns)

            # Make prediction
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)[:, 1]

            # Interpret result
            result = "The loan will default." if prediction[0] == 1 else "The loan will not default."
            # confidence = f"Confidence: {prediction_proba[0]:.4f}"

            return render_template("index.html", result=result)
            # return render_template("index.html", result=result, confidence=confidence)

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)