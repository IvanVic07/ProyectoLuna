import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import re

def convert_to_gb(storage):
    if 'TB' in storage:
        return float(storage.replace('TB', '')) * 1024
    elif 'GB' in storage:
        return float(storage.replace('GB', ''))
    else:
        return float(storage)

def extract_resolution(string):
    matches = re.findall(r'\d+', string)
    return [int(match) for match in matches] if matches else [0]  # Devuelve todos los números encontrados

# Cargar datos
data = pd.read_csv('/Users/jacquelinecarrillo/Desktop/8vo Semestre/BigData/Proyecto/new.csv', sep=";")
data["Almacenamiento_GB"] = data["Almacenamiento_GB"].apply(convert_to_gb)

# Definición de características
features = ['Almacenamiento_GB', 'RAM_GB', 'Precio', 'Disponibilidad', 'Resolucion_Camara', 
            'Capacidad_Bateria_mAh', 'Tamano_Pantalla', 'peso_g', 'Stock', 'Modelo']
data_subset = data[features].copy()
data_subset['Disponibilidad'] = (data_subset['Disponibilidad'] == 'Out of stock').astype(int)
data_subset['Resolucion_Camara'] = data_subset['Resolucion_Camara'].apply(extract_resolution)

# Expandir resolución de la cámara
max_len = data_subset['Resolucion_Camara'].apply(len).max()
column_names = [f'Resolucion_Camara_{i+1}' for i in range(max_len)]
camera_resolution = pd.DataFrame(data_subset['Resolucion_Camara'].tolist(), index=data_subset.index, columns=column_names)
data_subset = pd.concat([data_subset.drop('Resolucion_Camara', axis=1), camera_resolution], axis=1)

# Codificar variables categóricas
data_subset = pd.get_dummies(data_subset, columns=['Modelo'], drop_first=True)

# Preparar datos para entrenamiento
X = data_subset.drop('Precio', axis=1)
y = data_subset['Precio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Asumiendo que se ha realizado una correcta expansión y codificación de características

# Imputador para valores NaN
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Normalizar características numéricas después de la imputación
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Entrenar modelo
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print("Error cuadrático medio del modelo:", mse)

# Preparar la imputación y escala para la predicción de un nuevo producto
producto_nuevo = [128, 4, 1, 16, 5000, 6.5, 200, 150, 7, 10, 0] + [0] * (len(X_train.columns) - 11)
producto_nuevo_imputed = imputer.transform([producto_nuevo])  # Imputar primero
producto_nuevo_scaled = scaler.transform(producto_nuevo_imputed)  # Luego escalar
precio_predicho = model.predict(producto_nuevo_scaled)
print(f"Se espera que el precio del producto sea de aproximadamente ${precio_predicho[0]:.2f}.")