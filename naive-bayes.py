import numpy as np
import kagglehub as kh
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Implementación de Naive Bayes para clasificación de texto. Recomendado para detectar
# spam e identificar el idioma de un texto.
# El algoritmo asume que las características son independientes entre sí.

def naive_bayes(X_train, y_train, X_test):
    # Inicializar las probabilidades
    classes = np.unique(y_train)
    probs = {}
    for c in classes:
        probs[c] = {}
        # Calcular la probabilidad inicial
        # (contar el número de instancias de cada clase y dividirlo por el total)
        probs[c]['initial'] = np.sum(y_train == c) / len(y_train)
        print(f'Class {c} initial probability: {probs[c]["initial"]:.2f}')

        # Calcular la probabilidad condicional
        # (contar el número de veces que aparece cada palabra en cada clase y dividirlo por el total)
        probs[c]['cond'] = {}
        for i, word in enumerate(X_train):
            if c not in probs[c]['cond']:
                probs[c]['cond'][i] = {}
            for j, w in enumerate(word):
                if w not in probs[c]['cond'][i]:
                    probs[c]['cond'][i][w] = 0
                probs[c]['cond'][i][w] += 1
    # Calcular la probabilidad condicional
    # (dividir el número de veces que aparece cada palabra en cada clase por el total)
    # sumar 1 para evitar los ceros
    for c in classes:
        for i in probs[c]['cond']:
            for w in probs[c]['cond'][i]:
                probs[c]['cond'][i][w] = (probs[c]['cond'][i][w] + 1) / (np.sum(X_train == w) + 1)

    # Clasificar los datos de prueba
    # (calcular la probabilidad de cada clase para cada palabra y seleccionar la clase con la probabilidad más alta)
    predictions = []
    for i, word in enumerate(X_test):
        max_prob = -1
        max_class = None
        for c in classes:
            prob = probs[c]['initial']
            for j, w in enumerate(word):
                if w in probs[c]['cond'][j]:
                    prob *= probs[c]['cond'][j][w]
            if prob > max_prob:
                max_prob = prob
                max_class = c
        predictions.append(max_class)
    return predictions


# Descargar ultima version del dataset para deteccion de spam
path = kh.dataset_download("shantanudhakadd/email-spam-detection-dataset-classification") + '/spam.csv'

# Cargar el dataset
data = pd.read_csv(path, encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'text']
data.head()

# Preprocesar el dataset
data['text'] = data['text'].str.lower().str.split()
X = data['text'].values
y = data['label'].values

# Dividir el conjunto de datos en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Clasificar los puntos de prueba
y_pred = naive_bayes(X_train, y_train, X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
