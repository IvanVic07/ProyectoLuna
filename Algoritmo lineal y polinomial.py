import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Definir la función convert_to_gb
def convert_to_gb(storage):
    if 'TB' in storage:
        return float(storage.replace('TB', '')) * 1024  # Convertir de TB a GB
    elif 'GB' in storage:
        return float(storage.replace('GB', ''))  # Mantener los valores en GB
    else:
        return float(storage) 

# Cargar los datos
data = pd.read_csv('/Users/jacquelinecarrillo/Desktop/8vo Semestre/BigData/Proyecto/new.csv', sep=";")  # Especificar el separador como punto y coma

# Aplicar la función convert_to_gb a la columna "Almacenamiento_GB"
data["Almacenamiento_GB"] = data["Almacenamiento_GB"].apply(convert_to_gb)

# Separar las características y la variable dependiente
X = data.drop(columns=["Precio"])
y = data["Precio"]

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir las columnas categóricas y numéricas
categorical_cols = ['Marca', 'Modelo', 'Color', 'Disponibilidad', 'Sistema_Operativo', 'Clasificacion_IP','Fecha_Lanzamiento','Resolucion_Camara', 'Dimensiones_mm']
numeric_cols = ['Almacenamiento_GB', 'RAM_GB', 'Precio_USD', 'Capacidad_Bateria_mAh', 'Tamano_Pantalla', 'peso_g', 'Stock']



# Crear un transformador para las columnas categóricas
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Crear un transformador para las columnas numéricas
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combinar los transformadores
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Crear un pipeline con el preprocesamiento y el modelo original (lineal)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', LinearRegression())])

# Entrenar el modelo original (lineal)
pipeline.fit(X_train, y_train)

# Realizar predicciones con el modelo original (lineal)
y_pred = pipeline.predict(X_test)

# Calcular el error cuadrático medio para el modelo original (lineal)
mse = mean_squared_error(y_test, y_pred)
print("Error cuadrático medio:", mse)

# Crear un pipeline con el preprocesamiento, la transformación polinomial y el modelo (polinomial)
pipeline_poly = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('polynomial_features', PolynomialFeatures(degree=2)),
    ('regressor', LinearRegression())
])

# Entrenar el modelo de regresión polinomial
pipeline_poly.fit(X_train, y_train)

# Realizar predicciones con el modelo de regresión polinomial
y_pred_poly = pipeline_poly.predict(X_test)

# Calcular el error cuadrático medio para el modelo de regresión polinomial
mse_poly = mean_squared_error(y_test, y_pred_poly)
print("Error cuadrático medio para regresión polinomial:", mse_poly)
