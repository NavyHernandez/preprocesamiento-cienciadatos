import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocesar_datos(df):
    # 1. Eliminación de duplicados [cite: 24]
    df = df.drop_duplicates()
    
    # 2. Gestión de valores nulos [cite: 24, 31]
    # Rellenar nulos numéricos con la media
    for col in df.select_dtypes(include=['number']).columns:
        df[col] = df[col].fillna(df[col].mean())
    
    # 3. Codificación de variables categóricas [cite: 24, 31]
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col].astype(str))
        
    # 4. Normalización de datos numéricos [cite: 24, 31]
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = scaler.fit_transform(df[num_cols])
        
    return df

if __name__ == "__main__":
    print("Módulo de preprocesamiento para Ciencia de Datos cargado exitosamente.")
