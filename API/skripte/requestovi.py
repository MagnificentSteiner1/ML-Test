from fastapi import APIRouter
from .scheme import Input_Customer, Output_Customer
import requests



#skripta koja samo salje requestove

# featuri koji se koriste
# TotalCharges        0.295592
# tenure              0.103019
# Contract            0.080427
# MonthlyCharges      0.076547
# PaymentMethod       0.059105

router = APIRouter(prefix="/API/skripte/requestovi", tags=["Requestovi"])

@router.post("/predict", response_model = Output_Customer)
def predict(customer: Input_Customer):

    response = requests.post(
        "http://ml:5000/predict",
        json=customer.model_dump()
    )

    return response.json()
