# Importación de bibliotecas necesarias
import pandas  # Para manejar datos en formato tabular (DataFrames)
import numpy  # Para operaciones numéricas
from scipy.stats import norm  # Para calcular distribuciones normales

# Definición de la clase NaiveBayes
class NaiveBayes:

    # Constructor de la clase
    def __init__(self, filename, porcentaje_entrenamiento, clase_objetivo):
        # Inicialización de atributos
        self.filename = filename  # Nombre del archivo CSV con los datos
        self.porcentaje_entrenamiento = porcentaje_entrenamiento  # Porcentaje de datos para entrenamiento
        self.porcentaje_prueba = 100 - porcentaje_entrenamiento  # Porcentaje de datos para prueba
        self.dataset = pandas.read_csv(self.filename)  # Carga del conjunto de datos desde el archivo CSV
        # Selección de un subconjunto aleatorio para entrenamiento
        self.dataset_entrenamiento = self.dataset.sample(frac=(porcentaje_entrenamiento / 100))
        # Creación del conjunto de prueba excluyendo las instancias de entrenamiento
        self.dataset_prueba = self.dataset.drop(self.dataset_entrenamiento.index)
        self.clase_objetivo = clase_objetivo  # La columna que se desea predecir

    # Método para entrenar el modelo Naive Bayes
    def entrenar(self):
        print(f'\nEntrenando con el {self.porcentaje_entrenamiento}% de las instancias...\n')
        print(self.dataset_entrenamiento)

        # Extraer la columna de la clase objetivo
        columna_clase = self.dataset_entrenamiento[self.clase_objetivo]

        # Obtener los posibles valores de la clase objetivo
        posibles_valores_clase = columna_clase.unique()

        # Creación de una tabla de frecuencia
        print('\nGenerando tabla de frecuencia...')
        tabla_frecuencia = {}

        # Recorre cada atributo en las columnas del conjunto de entrenamiento
        for atributo in self.dataset_entrenamiento.columns:
            tabla_frecuencia[atributo] = {}

            # Crear una tupla que combina los valores del atributo y la clase
            valor_clase = tuple(zip(self.dataset_entrenamiento[atributo], columna_clase))

            if atributo == self.clase_objetivo:
                # Si el atributo es la clase objetivo, calcular la frecuencia de cada clase
                tabla_frecuencia[atributo]['Total'] = 0
                for clase in columna_clase:
                    if clase not in tabla_frecuencia[atributo].keys():
                        tabla_frecuencia[atributo][clase] = {}
                        for posible_valor_clase in posibles_valores_clase:
                            tabla_frecuencia[atributo][clase][posible_valor_clase] = 0
                    tabla_frecuencia[atributo]['Total'] += 1
                    tabla_frecuencia[atributo][clase][clase] += 1
            elif self.dataset_entrenamiento[atributo].dtypes == 'int64' or self.dataset_entrenamiento[atributo].dtypes == 'float64':
                # Si el atributo es numérico, guardar todos los valores numéricos para cada clase
                for posible_valor_clase in posibles_valores_clase:
                    tabla_frecuencia[atributo][posible_valor_clase] = {}
                for valor, clase in valor_clase:
                    tabla_frecuencia[atributo][clase][valor] = 1
            else:
                # Si el atributo es categórico, calcular la frecuencia de cada valor para cada clase
                tabla_frecuencia[atributo]['Total'] = {}
                for posible_valor_clase in posibles_valores_clase:
                    tabla_frecuencia[atributo]['Total'][posible_valor_clase] = len(
                        self.dataset_entrenamiento[atributo].unique())
                for valor, clase in valor_clase:
                    if valor not in tabla_frecuencia[atributo].keys():
                        tabla_frecuencia[atributo][valor] = {}
                        for posible_valor_clase in posibles_valores_clase:
                            tabla_frecuencia[atributo][valor][posible_valor_clase] = 0
                    tabla_frecuencia[atributo]['Total'][clase] += 1
                    tabla_frecuencia[atributo][valor][clase] += 1

        # Creación de una tabla de verosimilitud
        print('\nGenerando tabla de verosimilitud...')
        self.tabla_verosimilitud = {}

        # Recorre la tabla de frecuencia para calcular la verosimilitud
        for atributo in tabla_frecuencia:
            self.tabla_verosimilitud[atributo] = {}
            if atributo == self.clase_objetivo:
                # Verosimilitud de cada clase
                for clase in tabla_frecuencia[atributo]:
                    if clase != 'Total':
                        verosimilitud = tabla_frecuencia[atributo][clase][clase] / tabla_frecuencia[atributo]['Total']
                        self.tabla_verosimilitud[atributo][clase] = verosimilitud
            elif self.dataset_entrenamiento[atributo].dtypes == 'int64' or self.dataset_entrenamiento[
                atributo].dtypes == 'float64':
                # Verosimilitud de los atributos numéricos
                for clase in tabla_frecuencia[atributo]:
                    self.tabla_verosimilitud[atributo][clase] = {}
                    valores_numericos = []
                    for valor in tabla_frecuencia[atributo][clase]:
                        valores_numericos.append(valor)
                    self.tabla_verosimilitud[atributo][clase]['Media'] = numpy.mean(valores_numericos) # Media de los valores del atributo y clase.
                    self.tabla_verosimilitud[atributo][clase]['DesvEst'] = numpy.std(valores_numericos) # Desviación estándar de los valores del atributo y clase.
            else:
                # Verosimilitud de cada valor de cada atributo categórico
                for valor in tabla_frecuencia[atributo]:
                    if valor != 'Total':
                        self.tabla_verosimilitud[atributo][valor] = {}
                        for clase in tabla_frecuencia[atributo][valor]:
                            verosimilitud = tabla_frecuencia[atributo][valor][clase] / \
                                            tabla_frecuencia[atributo]['Total'][clase]
                            self.tabla_verosimilitud[atributo][valor][clase] = verosimilitud

    # Método para probar el modelo Naive Bayes
    def probar(self):
        print(f'\nProbando con el {self.porcentaje_prueba}% de las instancias...\n')
        print(self.dataset_prueba)
        columna_clase = self.dataset_prueba[self.clase_objetivo]
        posibles_valores_clase = columna_clase.unique()

        # Cálculo de las probabilidades posteriores para cada instancia de prueba
        print('\nCalculando probabilidades posteriores...')
        probabilidades_posteriores = {}  # Almacena las probabilidades posteriores de las instancias de prueba
        clases_mayor_probabilidad = {}  # Almacena la clase más probable para cada instancia de prueba

        # Iteración sobre las instancias de prueba
        for index in self.dataset_prueba.index:
            probabilidades_posteriores[index] = {}
            for posible_valor_clase in posibles_valores_clase:
                probabilidad_posterior = 1  # Variable para calcular la probabilidad posterior
                # Iteración sobre los atributos de la instancia
                for atributo in self.dataset_prueba.columns:
                    if atributo == self.clase_objetivo:
                        probabilidad_posterior = probabilidad_posterior * self.tabla_verosimilitud[atributo][
                            posible_valor_clase]
                    elif self.dataset_entrenamiento[atributo].dtypes == 'int64' or self.dataset_entrenamiento[
                        atributo].dtypes == 'float64':
                        valor = self.dataset_prueba.at[index, atributo]
                        # Cálculo de la probabilidad usando la función de densidad de probabilidad de una distribución normal
                        funcion_densidad_probabilidad = norm.pdf(
                                valor,
                                loc=self.tabla_verosimilitud[atributo][posible_valor_clase]['Media'],
                                scale=self.tabla_verosimilitud[atributo][posible_valor_clase]['DesvEst']
                            )
                        probabilidad_posterior = probabilidad_posterior * funcion_densidad_probabilidad
                    else:
                        valor = self.dataset_prueba.at[index, atributo]
                        probabilidad_posterior = probabilidad_posterior * self.tabla_verosimilitud[atributo][valor][
                            posible_valor_clase]
                # Se almacena la probabilidad posterior de la clase
                probabilidades_posteriores[index][posible_valor_clase] = probabilidad_posterior
            mayor_probabilidad = 0
            clases_mayor_probabilidad[index] = ''
            # Selección de la clase con mayor probabilidad para esta instancia de prueba
            for posible_valor_clase in posibles_valores_clase:
                if probabilidades_posteriores[index][posible_valor_clase] > mayor_probabilidad:
                    mayor_probabilidad = probabilidades_posteriores[index][posible_valor_clase]
                    clases_mayor_probabilidad[index] = posible_valor_clase

        # Análisis de resultados (Matriz de confusión)
        print('\nGenerando matriz de confusión...')
        # Se crea un DataFrame para hacer la matriz con los nombres de las clases como filas y columnas.
        matriz_confusion = pandas.DataFrame(0, index=posibles_valores_clase, columns=posibles_valores_clase)

        # Llenado de la matriz de confusión
        for index in self.dataset_prueba.index:
            clase_esperada = self.dataset_prueba.at[index, self.clase_objetivo]  # Clase real
            clase_estimada = clases_mayor_probabilidad[index]  # Clase estimada por el modelo
            matriz_confusion.at[clase_esperada, clase_estimada] += 1

        print('\nMatriz de confusión:')
        print(matriz_confusion)

        # Cálculo del desempeño del modelo
        print('\nDesempeño del modelo')

        # Exactitud
        numerador_exactitud = 0
        denominador_exactitud = 0
        # Iteración sobre las clases en la matriz de confusión
        for columna in posibles_valores_clase:
            for fila in posibles_valores_clase:
                if columna == fila:
                    # Si la clase de la fila y la columna son iguales son predicciones correctas y se suma al numerador.
                    numerador_exactitud = numerador_exactitud + matriz_confusion.at[fila, columna]
                # Se suma al numerador todos los valores de la matriz
                denominador_exactitud = denominador_exactitud + matriz_confusion.at[fila, columna]

        # Se calcula la exactitud dividiendo el total de predicciones correctas entre el total de predicciones.
        exactitud = numerador_exactitud / denominador_exactitud
        print(f'Exactitud: {exactitud}')

        return matriz_confusion

    # Método para calcular la precisión para cada clase
    def calcular_precision(self, matriz_confusion):
        print('\nCalculando precisión para cada clase...')
        posibles_valores_clase = matriz_confusion.index
        precisiones = {}

        # Cálculo de la precisión para cada clase
        for posible_valor_clase in posibles_valores_clase:
            TP = matriz_confusion.at[posible_valor_clase, posible_valor_clase]  # Verdaderos positivos (A)
            FN = matriz_confusion.loc[posible_valor_clase, :].sum() - TP  # Falsos negativos (B + C)
            precisiones[posible_valor_clase] = TP / (TP + FN)  # Precisión = A / (A + B + C)

        print('Precisión para cada clase:')
        for clase, precision in precisiones.items():
            print(f'Clase: {clase}, Precisión: {precision}')

        #             setosa    virginica    versicolor
        # setosa        A           B            C
        # virginica     D           E            F
        # versicolor    G           H            I

    # Método para calcular el recall para cada clase
    def calcular_recall(self, matriz_confusion):
        print('\nCalculando recall para cada clase...')
        posibles_valores_clase = matriz_confusion.index
        recalls = {}

        # Cálculo del recall para cada clase
        for posible_valor_clase in posibles_valores_clase:
            TP = matriz_confusion.at[posible_valor_clase, posible_valor_clase]  # Verdaderos positivos (A)
            FP = matriz_confusion.loc[:, posible_valor_clase].sum() - TP  # Falsos positivos (D + G)
            recalls[posible_valor_clase] = TP / (TP + FP)  # Recall = A / (A + D + G)

        print('Recall para cada clase:')
        for clase, recall in recalls.items():
            print(f'Clase: {clase}, Recall: {recall}')


# Creación de una instancia de la clase NaiveBayes y ejecución de entrenamiento y prueba
naive_bayes = NaiveBayes('iris.data', 70, 'iris')
naive_bayes.entrenar()
matriz_confusion = naive_bayes.probar()

# Cálculo de la precisión y el recall para cada clase
naive_bayes.calcular_precision(matriz_confusion)
naive_bayes.calcular_recall(matriz_confusion)