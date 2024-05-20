import pickle 
from fastapi import FastAPI
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
from pydantic import BaseModel
from joblib import load
from fastapi.middleware.cors import CORSMiddleware

class House(BaseModel):
    Location: str 
    wifi: int
    backupPower: int
    Stove:int
    Fridge:int
    separate_kitchen: int
    curfew: int 
    distance: float
    visitors:int 
    Shelves:str
    Water_tank: int
    maid: int
    gas_stove: int
    gyser:int 
    gender: str
    swimming_pool:int
    per_room:int
    beds:int
    security: int
    meals:int
    


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=False,  # Set to True if you need to send cookies across origins
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, etc.)
    allow_headers=["*"],  # Allow all headers sent by the client
)

with open('xg_boost.pkl', 'rb') as f:
    model = pickle.load(f)
   
with open('transformer.pkl', 'rb') as f:
    transformer = pickle.load(f)


@app.post('/predict')
async def predict(data: House):
    point = {
      "Location": data.Location, 
      "wifi": data.wifi,
      "backupPower": data.backupPower,
       "Stove": data.Stove,
      "Fridge": data.Fridge,
      "separate_kitchen": data.separate_kitchen,
      "curfew": data.curfew,
      "distance": data.distance,
      "visitors": data.visitors,
      "Shelves": data.Shelves,
      "Water_tank":data.Water_tank,
      "maid": data.maid,
      "gas_stove": data.gas_stove,
      "gyser": data.gyser,
      "gender": data.gender,
      "swimming_pool": data.swimming_pool,
      "per_room": data.per_room,
      "beds": data.beds,
      "security": data.security,
      "meals": data.meals      
  } 

    df =pd.DataFrame( pd.Series(point)).T 
    df_transformed =  transformer.transform(df)

    pred = model.predict(df_transformed)

    return pred.tolist()[0]