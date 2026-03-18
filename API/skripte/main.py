from fastapi import FastAPI
from .requestovi import router as requestovi_router

#Skripta za predvidjanje ban rate-a, ako se unese ime championa i pozicija
app = FastAPI()
app.include_router(requestovi_router)