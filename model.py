import pandas as pd
import numpy as np
import random
import pickle
from sklearn.linear_model import Lasso
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error as MSE


random.seed(42)
np.random.seed(42)

df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
df_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')

def to_int():
    df_train['engine'] = df_train['engine'].astype('int')
    df_test['engine'] = df_test['engine'].astype('int')
    

def to_float():
    df_train['max_power'] = df_train['max_power'].astype('str')
    df_train['max_power'] = df_train['max_power'].str.replace('bhp', '')
    df_train['max_power'] = pd.to_numeric(df_train['max_power'], errors='coerce')
    df_train['max_power'] = df_train['max_power'].astype('float64')

    df_test['max_power'] = df_test['max_power'].astype('str')
    df_test['max_power'] = df_test['max_power'].str.replace('bhp', '')
    df_test['max_power'] = pd.to_numeric(df_test['max_power'], errors='coerce')
    df_test['max_power'] = df_test['max_power'].astype('float64')

    df_train['engine'] = df_train['engine'].astype('str')
    df_train['engine'] = df_train['engine'].str.replace('CC', '')
    df_train['engine'] = pd.to_numeric(df_train['engine'], errors='coerce')
    df_train['engine'] = df_train['engine'].astype('float64')

    df_test['engine'] = df_test['engine'].astype('str')
    df_test['engine'] = df_test['engine'].str.replace('CC', '')
    df_test['engine'] = pd.to_numeric(df_test['engine'], errors='coerce')
    df_test['engine'] = df_test['engine'].astype('float64')

    df_train['mileage'] = df_train['mileage'].astype('str')
    df_train['mileage'] = df_train['mileage'].str.replace('kmpl', '')
    df_train['mileage'] = pd.to_numeric(df_train['mileage'], errors='coerce')
    df_train['mileage'] = df_train['mileage'].astype('float64')

    df_test['mileage'] = df_test['mileage'].astype('str')
    df_test['mileage'] = df_test['mileage'].str.replace('kmpl', '')
    df_test['mileage'] = pd.to_numeric(df_test['mileage'], errors='coerce')
    df_test['mileage'] = df_test['mileage'].astype('float64')

def nan_to_median():
    for column in df_train.select_dtypes(['float']).columns:
        median = df_train[column].median()
        df_train[column] = df_train[column].fillna(median) 

    for column in df_test.select_dtypes(['float']).columns:
        median = df_test[column].median()
        df_test[column] = df_test[column].fillna(median) 

df_train = df_train.drop_duplicates()
to_float()
nan_to_median()
df_train = df_train.drop('torque', axis=1)
df_test = df_test.drop('torque', axis=1)
df_train = df_train.drop('seats', axis=1)
df_test = df_test.drop('seats', axis=1)

df_train_with_cat = df_train.drop('name', axis=1)
df_test_with_cat = df_test.drop('name', axis=1)
to_int()

# cat_features = ['fuel', 'seller_type', 'transmission', 'owner']
# num_features = ['year', 'selling_price', 'km_driven', 'mileage', 'engine', 'max_power']
# ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# df_train_with_cat_ohe = ohe.fit_transform(df_train_with_cat[cat_features])
# df_test_with_cat_ohe = ohe.fit_transform(df_test_with_cat[cat_features])
# df_train_with_cat = pd.concat([pd.DataFrame(df_train_with_cat, columns=num_features),
#                                     pd.DataFrame(df_train_with_cat_ohe, columns=ohe.get_feature_names_out())
#                                    ], axis=1)
# df_test_with_cat = pd.concat([pd.DataFrame(df_test_with_cat, columns=num_features),
#                                     pd.DataFrame(df_test_with_cat_ohe, columns=ohe.get_feature_names_out())
#                                    ], axis=1)
# df_train_with_cat = df_train_with_cat.dropna()

df_train_with_cat = pd.get_dummies(df_train_with_cat, columns =['fuel', 'seller_type', 'transmission', 'owner'])
df_test_with_cat = pd.get_dummies(df_test_with_cat, columns =['fuel', 'seller_type', 'transmission', 'owner'])

y_train_with_cat = df_train_with_cat['selling_price']
X_train_with_cat = df_train_with_cat.drop('selling_price', axis=1)
y_test_with_cat = df_test_with_cat['selling_price']
X_test_with_cat = df_test_with_cat.drop('selling_price', axis=1)

model_l1 = Lasso(alpha=0.01)

model_l1.fit(X_train_with_cat, y_train_with_cat)

pred_test_with_cat = model_l1.predict(X_test_with_cat)
mse_test = MSE(y_test_with_cat, pred_test_with_cat)
r2_test = r2_score(y_test_with_cat, pred_test_with_cat)
print('MSE:', mse_test)
print('r2:', r2_test)

model_filename = 'model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(model_l1, model_file)
