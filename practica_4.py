import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#!Importamos la clase de la red neuronal
from mpl import NeuralNet
    
#!Funcion para leer el archivo csv
def readFile(f):

    data = pd.read_csv(f)

    #*El archivo tiene cuatro columnas como entradas y tres de salidas
    inputs = data.iloc[:,:4]
    outputs = data.iloc[:,4:]

    #*Usando la libreria de sklearn vamos a dividir el set de datos en 80% para train y 20% para test
    inputTrain, inputTest, outputTrain, outputTest = train_test_split(inputs, outputs, test_size=0.2, random_state=2)

    print(f"{inputTrain.shape}, {inputTest.shape}, {outputTrain.shape}, {outputTest.shape}")

    #*Regresa todos los sets creados
    return inputs, outputs, inputTrain, inputTest, outputTrain, outputTest

#!Creamos las variables para guardar inputs y los tensores
#*Leemos el archivo
inputs, outputs, inputTrain, inputTest, outputTrain, outputTest = readFile("irisbin.csv")

print(inputTest)

#*Variables para guardar los tensores de la red neuronal
#*Los tensores se van a configurar para que los ejecute el cpu
tenInputTrain = torch.from_numpy(inputTrain.values).float().to("cpu")
tenInputTest = torch.from_numpy(inputTest.values).float().to("cpu")
tenOutputTrain = torch.from_numpy(outputTrain.values).int().to("cpu")
tenOutputTest = torch.from_numpy(outputTest.values).int().to("cpu")

#!Funcion de perdida diseñada para este problema en especifico
def loss(predic, esperado):
    #*Usamos el error cuadratico medio
    return torch.mean((predic - esperado)**2)

#!Creamos las variables que seran utilizadas para la red neuronal
learningRate = 0.001
epochs = 1000

