from dataclasses import Field
from enum import Enum

from fastapi import FastAPI
from pydantic import BaseModel, Field

# featuri koji se koriste
# TotalCharges        0.295592
# tenure              0.103019
# Contract            0.080427
# MonthlyCharges      0.076547
# PaymentMethod
app = FastAPI()

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

class Output_Customer(BaseModel):
    churn: bool
    probability: float

    

