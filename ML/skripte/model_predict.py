import pandas as pd
from pathlib import Path
import joblib
from enum import Enum
from fastapi import FastAPI
from pydantic import BaseModel


app=FastAPI()

model_dir = Path(__file__).parent.parent / "churn_topn_original_pipeline.pkl"
churn_pipeline = joblib.load(model_dir)


class ContractEnum(str, Enum):
    month_to_month = 'Month-to-month'
    one_year = 'One year'
    two_year = 'Two year'
class PaymentMethodEnum(str, Enum):
    bank_transfer = 'Bank transfer (automatic)'
    credit_card = 'Credit card (automatic)'
    electronic_check = 'Electronic check'
    mailed_check = 'Mailed check'
class Input_Customer(BaseModel):
    TotalCharges: float
    tenure : int
    Contract: ContractEnum
    MonthlyCharges: float
    PaymentMethod: PaymentMethodEnum


def predict_churn_single(churn_pipeline,customer_dict):
    # Convert single dict to DataFrame
    input_df = pd.DataFrame([customer_dict])

    # Predict probability of positive class (churn = 'Yes')
    proba = churn_pipeline.predict_proba(input_df)[:, 1][0]

    # Predict label
    pred_label = churn_pipeline.predict(input_df)

    # Convert to boolean
    churn_bool = False if pred_label == "No" else True

    return {"churn": churn_bool, "probability": float(proba)}

@app.post("/predict")
def predict(customer: Input_Customer):
    customer_dict = customer.model_dump()  # correct way to get a dict from Pydantic model
    # convert Enum to string if needed
    customer_dict = {k: (v.value if isinstance(v, Enum) else v) for k, v in customer_dict.items()}
    return predict_churn_single(churn_pipeline, customer_dict)