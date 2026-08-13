import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv('crop_data.csv')
print("Dataset loaded:", df.shape)

le_region = LabelEncoder()
le_soil   = LabelEncoder()
le_crop   = LabelEncoder()

df['Region']    = le_region.fit_transform(df['Region'])
df['Soil_Type'] = le_soil.fit_transform(df['Soil_Type'])
df['Crop']      = le_crop.fit_transform(df['Crop'])

X = df[['Region', 'Soil_Type', 'Crop', 'Rainfall_mm', 'Temperature_C', 'Fertilizer_kg']]
y = df['Yield_per_hectare']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
print("Model trained successfully!")

with open('model.pkl', 'wb') as f:
    pickle.dump({
        'model':     model,
        'le_region': le_region,
        'le_soil':   le_soil,
        'le_crop':   le_crop
    }, f)

print("model.pkl saved!")
print("Crops  :", list(le_crop.classes_))
print("Soils  :", list(le_soil.classes_))
print("Regions:", list(le_region.classes_))
