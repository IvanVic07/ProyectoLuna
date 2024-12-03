import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

# Cargar el dataset
data = pd.read_csv('/Users/jacquelinecarrillo/Desktop/8vo Semestre/BigData/Proyecto/new.csv', sep=";")

# Preparar las características y la etiqueta
X = data[['Precio', 'Disponibilidad']]  # Asumiendo que quieres usar Precio y Disponibilidad como características
y = data['Stock']

# Codificar la variable categórica 'Disponibilidad'
le = preprocessing.LabelEncoder()
X.loc[:, 'Disponibilidad'] = le.fit_transform(data['Disponibilidad'])

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Error cuadrático medio del modelo:", mse)

# Predicción
precio = 800  # Precio del producto a predecir
disponibilidad = 'In stock'  # Disponibilidad del producto a predecir

# Codificar la disponibilidad para la predicción
disponibilidad_codificada = le.transform([disponibilidad])[0]

# Crear un DataFrame para la predicción
df_pred = pd.DataFrame([[precio, disponibilidad_codificada]], columns=['Precio', 'Disponibilidad'])

# Realizar la predicción
stock_prediccion = model.predict(df_pred)
print(f"Se espera que el stock del producto sea de aproximadamente {stock_prediccion[0]:.0f} unidades.")

# Crear un DataFrame para mostrar los resultados de las predicciones y los valores reales
results = pd.DataFrame({
    'Precio': X_test['Precio'],
    'Disponibilidad': X_test['Disponibilidad'],
    'Stock real': y_test,
    'Stock predicho': y_pred
})

# Mostrar los resultados en una tabla
print(results.head(10))  # Mostrar solo las primeras 10 filas
