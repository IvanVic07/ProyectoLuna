import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Cargar el dataset
data = pd.read_csv('/Users/jacquelinecarrillo/Desktop/8vo Semestre/BigData/Proyecto/new.csv', sep=";")

# Preparar las características y la etiqueta
# Asumiremos que quieres predecir la 'Disponibilidad' usando 'Precio' y 'Almacenamiento_GB' como características
X = data[['Precio', 'Almacenamiento_GB']]
y = data['Disponibilidad']

# Codificar la variable categórica 'Disponibilidad'
le = LabelEncoder()
y = le.fit_transform(y)

# Convertir el almacenamiento a un valor numérico, si es necesario
X['Almacenamiento_GB'] = pd.to_numeric(X['Almacenamiento_GB'], errors='coerce').fillna(0)

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Realizar predicciones sobre el conjunto de prueba
y_pred = knn.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)

# Realizar una predicción con nuevos datos
nuevo_precio = 800  # Precio del producto a predecir
nuevo_almacenamiento = 128  # Almacenamiento del producto a predecir

# Crear un DataFrame para la predicción
nueva_prediccion = pd.DataFrame([[nuevo_precio, nuevo_almacenamiento]], columns=['Precio', 'Almacenamiento_GB'])

# Predecir la disponibilidad usando el modelo KNN
disponibilidad_predicha = knn.predict(nueva_prediccion)
disponibilidad_texto = le.inverse_transform(disponibilidad_predicha)
print(f"Se espera que la disponibilidad del producto sea: {disponibilidad_texto[0]}")