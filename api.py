import pickle 
from fastapi import FastAPI
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
from pydantic import BaseModel
from joblib import load

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

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
   
with open('transformer.pkl', 'rb') as f:
    transformer = pickle.load(f)

items = {0: {0: ['location', 'pleasant']}, 1:{0: ['wifi',1]}, 2: {0: ['gyser', 1]}}

# list = [[]]

# for i in range(0,2):
#     # print(items[i][0][0])

#     # print(items[i][0][1])
#     list[0].append(items[i][0][1])

# print(list)    
import numpy as np     

# predict = load('test.joblib')     

# @app.post('/predict')
# async def predict(data: House):

    # point 
#     point = pd.DataFrame({
#       "Location": [data.Location], 
#       "wifi": [[data.wifi]],
#       "backupPower": [[data.backupPower]],
#        "Stove": [[data.Stove]],
#       "Fridge": [[data.Fridge]],
#       "separate_kitchen": [[data.separate_kitchen]],
#       "curfew": [[data.curfew]],
#       "distance": [[data.distance]],
#       "visitors": [[data.visitors]],
#       "Shelves": [[data.Shelves]],
#       "Water_tank":[[data.Water_tank]],
#       "maid": [[data.maid]],
#       "gas_stove": [[data.gas_stove]],
#       "gyser": [[data.gyser]],
#       "gender": [[data.gender]],
#       "swimming_pool": [[data.swimming_pool]],
#       "per_room": [[data.per_room]],
#       "beds": [[data.beds]],
#       "security": [[data.security]],
#       "meals": [[data.meals]]      
#   })
        # point = np.array([['mt_pleasant',0,0,0,0,0,0,2,0,'yes',0,0,0,0,'both',0,5,0,0,0]])
    
#     augah = {
#     "Location": {
#         "0": "mt_pleasant",
#     },
#     "wifi": {
#         "0": 
#             1
        
#     },
#     "backupPower": {
#         "0": 
#             1
        
#     },
#     "Stove": {
#         "0": 
#             1
        
#     },
#     "Fridge": {
#         "0": 
#             0
        
#     },
#     "separate_kitchen": {
#         "0": 
#             1
        
#     },
#     "curfew": {
#         "0": 
#             0
        
#     },
#     "distance": {
#         "0": 
#             2.456
        
#     },
#     "visitors": {
#         "0": 
#             1
        
#     },
#     "Shelves": {
#         "0": 
#             "yes"
        
#     },
#     "Water_tank": {
#         "0": 
#             1
    
#     },
#     "maid": {
#         "0": 
#             1
        
#     },
#     "gas_stove": {
#         "0": 
#             1
        
#     },
#     "gyser": {
#         "0": 
#             1
        
#     },
#     "gender": {
#         "0": 
#             "both"
        
#     },
#     "swimming_pool": {
#         "0": 
#             0
        
#     },
#     "per_room": {
#         "0": 
#             5
        
#     },
#     "beds": {
#         "0": 
#             0
        
#     },
#     "security": {
#         "0": 
#             1
        
#     },
#     "meals": {
#         "0": 
#                       1
     
#     }
# } 
    
         
    # point = pd.DataFrame(augah)

    
    

    # point = pd.DataFrame(data)
    # reshaped_data = [
    # [
    #     "mt_pleasant",
    #     1,
    #     1,
    #     1,
    #     0,
    #     1,
    #     0,
    #     2.456,
    #     1,
    #     "yes",
    #     1,
    #     1,
    #     1,
    #     1,
    #     "both",
    #     0,
    #     5,
    #     0,
    #     1,
    #     1
    # ]
# ]

    




      
# reshape = {"location": "mt", 'wifi': 0}
reshape = {"Location": "mt_pleasant", 
      "wifi": 1,
      "backupPower": 1,
       "Stove": 0,
      "Fridge": 1,
      "separate_kitchen": 0,
      "curfew": 1,
      "distance": 2,
      "visitors": 1,
      "Shelves": 'yes',
      "Water_tank":1,
      "maid": 0,
      "gas_stove": 1,
      "gyser": 0,
      "gender": 'boys',
      "swimming_pool": 0,
      "per_room": 1,
      "beds": 1,
      "security": 0,
      "meals": 0}
augah =pd.DataFrame( pd.Series(reshape)).T
print(transformer.transform(augah))

print(model.predict(transformer.transform(augah)))


