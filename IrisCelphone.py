import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Cargar datos desde el archivo CSV
df = pd.read_csv("/Users/jacquelinecarrillo/Desktop/8vo Semestre/BigData/Proyecto/Celulares2.csv", delimiter=";")

# Seleccionar solo las variables numéricas relevantes para K-means
df_numeric = df[["Almacenamiento_GB", "RAM_GB", "Precio_USD", "Capacidad_Bateria_mAh", "Tamano_Pantalla", "peso_g"]]

# Convertir el DataFrame a una matriz numpy para K-means
x = df_numeric.values

# Encontrar el valor óptimo de clusters utilizando el método del codo
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# Graficar el método del codo
plt.plot(range(1, 11), wcss)
plt.title("El método del codo")
plt.xlabel("Número de clusters")
plt.ylabel("WCSS")
plt.show()

# Aplicar K-means con el número óptimo de clusters
num_clusters = 3  # Número de clusters óptimo seleccionado
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, max_iter=300, random_state=0)
y_kmeans = kmeans.fit_predict(x)

# Agregar la columna de clusters al DataFrame original
df['Cluster'] = y_kmeans

# Visualizar los clusters en un gráfico de dispersión
sns.scatterplot(data=df, x='Almacenamiento_GB', y='RAM_GB', hue='Marca', style='Cluster', palette='viridis')
plt.xlabel('Almacenamiento (GB)')
plt.ylabel('RAM (GB)')
plt.title('Gráfico de dispersión de Almacenamiento vs RAM por Marca y Cluster')
plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1))
plt.show()
