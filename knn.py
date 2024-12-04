import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Implementación de KNN (K-Nearest Neighbors) para clasificación. Recomendado para problemas de
# clasificación con dimensiones bajas.
# Podemos tener como parámetro adicional la distancia que se va a emplear.


def distances(X_train, X_test):
    dist = np.zeros((X_test.shape[0], X_train.shape[0]))
    for i in range(X_test.shape[0]):
        for j in range(X_train.shape[0]):
            dist[i, j] = np.linalg.norm(X_test[i] - X_train[j])
    return dist


# Definir la función KNN
def knn(X_train, y_train, X_test, k):
    dist = distances(X_train, X_test)
    # Obtener los índices de los k vecinos más cercanos
    idx = np.argsort(dist, axis=1)[:, :k]
    knn_labels = y_train[idx]
    # Predecir la clase de los puntos de prueba basado en la clase que predomina de los vecinos más cercanos
    y_pred = np.array([np.bincount(labels).argmax() for labels in knn_labels])
    return y_pred


# region Pruebas


# Cargar el conjunto de datos (dataset Iris que se usa para clasificación)
iris = load_iris()
X = iris.data[:, :2]  # Usar solo las dos primeras características para visualización
y = iris.target

# Dividir el conjunto de datos en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Clasificar los puntos de prueba
k = 3
y_pred = knn(X_train, y_train, X_test, k)

# Graficar los resultados
plt.figure(figsize=(12, 8))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', label='Puntos de training')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', edgecolor='k', marker='*', s=200,
            label='Puntos de testing')
plt.legend()
plt.title('Resultados de KNN con k = ' + str(k))
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.show()
