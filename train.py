import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 1. LOAD DATA & CLEANING AWAL
df = pd.read_csv('laptops.csv')

df = df[df['Status'] == 'New']

df['GPU'] = df['GPU'].fillna('None/Integrated')
df['Storage type'] = df['Storage type'].fillna('Unknown')

df['Price_USD'] = df['Final Price'] * 1.08 

# 2. EXPLORATORY DATA ANALYSIS (EDA)
print("EDA")
print(f"Jumlah baris dan kolom: {df.shape}")
print("\nInfo Dataset:")
print(df.info())

print("\nCek Data Kosong (Missing Values):")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Setting visualisasi
sns.set_theme(style="whitegrid")

# Grafik 1: Distribusi Harga
plt.figure(figsize=(10, 5))
sns.histplot(df['Price_USD'], bins=30, kde=True, color='blue')
plt.title('Distribusi Harga Laptop Pasar (USD)')
plt.xlabel('Harga (USD)')
plt.ylabel('Frekuensi')
plt.show()

# Grafik 2: Rata-rata Harga Berdasarkan Merek (Top 10)
plt.figure(figsize=(12, 6))
top_brands = df['Brand'].value_counts().head(10).index
sns.barplot(data=df[df['Brand'].isin(top_brands)], x='Brand', y='Price_USD', errorbar=None, palette='viridis')
plt.title('Rata-Rata Harga Laptop Berdasarkan Top 10 Merek Terpopuler')
plt.xticks(rotation=45)
plt.ylabel('Rata-rata Harga (USD)')
plt.show()

# Grafik 3: Pengaruh Kapasitas RAM terhadap Harga
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='RAM', y='Price_USD', palette='coolwarm')
plt.title('Distribusi Harga Berdasarkan Kapasitas RAM (GB)')
plt.xlabel('RAM (GB)')
plt.ylabel('Harga (USD)')
plt.show()

# 3. PREPROCESSING DATA
X = df[['Brand', 'Model', 'CPU', 'RAM', 'Storage', 'Storage type', 'GPU', 'Screen', 'Touch']]
y = df['Price_USD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['RAM', 'Storage', 'Screen']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['Brand', 'Model', 'CPU', 'Storage type', 'GPU', 'Touch']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='None/Integrated')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. TRAINING & EVALUASI MODEL
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
}

best_model = None
best_r2 = -float('inf')
best_model_name = ""

for name, model_algorithm in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model_algorithm)])
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"{name} -> R2 Score: {r2:.4f} | MAE: {mae:.2f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = pipeline
        best_model_name = name

print(f"\nModel terbaik adalah {best_model_name} dengan skor R2: {best_r2:.4f}")

# 5. EXPORT MODEL (.pkl)
joblib.dump(best_model, 'best_laptop_price_model.pkl')
print("File 'best_laptop_price_model.pkl' berhasil disimpan!")