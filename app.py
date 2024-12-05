from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from fastapi import File, UploadFile, HTTPException
import io
from fastapi.responses import StreamingResponse

app = FastAPI()

def process_attrs(df):
    # to float
    # df['max_power'] = df['max_power'].astype('str')
    df['max_power'] = df['max_power'].str.replace('bhp', '')
    df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')
    df['max_power'] = df['max_power'].astype('float64')

    # df['engine'] = df['engine'].astype('str')
    df['engine'] = df['engine'].str.replace('CC', '')
    df['engine'] = pd.to_numeric(df['engine'], errors='coerce')
    df['engine'] = df['engine'].astype('float64')

    
    # df['mileage'] = df['mileage'].astype('str')
    df['mileage'] = df['mileage'].str.replace('kmpl', '')
    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
    df['mileage'] = df['mileage'].astype('float64')

    # fill NA
    for column in df.select_dtypes(['float']).columns:
        median = df[column].median()
        df[column] = df[column].fillna(median) 

    df = df.drop('torque', axis=1)
    df = df.drop('seats', axis=1)
    df = df.drop('name', axis=1)

    df['engine'] = df['engine'].astype('int')
    df = df.drop('selling_price', axis=1)

    cat_features = ['fuel', 'seller_type', 'transmission', 'owner']
    num_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    df_ohe = ohe.fit_transform(df[cat_features])
    df = pd.concat([pd.DataFrame(df, columns=num_features),
                                    pd.DataFrame(df_ohe, columns=ohe.get_feature_names_out())
                                   ], axis=1)
    return df



class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

def model_lasso_item(x: dict) -> dict:
    with open("model.pkl", 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    x_df = pd.DataFrame(x, index=[0])
    res = loaded_model.predict(x_df)[0]
    return {"prediction": res}

def model_lasso_items(x: dict) -> dict:
    with open("model.pkl", 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    res = loaded_model.predict(x)
    return res

@app.post("/predict_item")
def predict_item(item: Item):
    train_columns = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'fuel_CNG',
       'fuel_Diesel', 'fuel_LPG', 'fuel_Petrol', 'seller_type_Dealer',
       'seller_type_Individual', 'seller_type_Trustmark Dealer',
       'transmission_Automatic', 'transmission_Manual', 'owner_First Owner',
       'owner_Fourth & Above Owner', 'owner_Second Owner',
       'owner_Test Drive Car', 'owner_Third Owner']
    df = pd.DataFrame(columns=train_columns)

    item = item.model_dump()
    item_df = pd.DataFrame(item, index=[0])
    processed_item = process_attrs(item_df)

    model_data = pd.concat([df, processed_item], axis=0)
    model_data = model_data.fillna(0)
    print(model_data)
    return model_lasso_item(model_data)


@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    processed_df = process_attrs(df)
    res = model_lasso_items(processed_df)
    df['prediction'] = res
    print(df)
    
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv"
                                 )
    
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    
    return response