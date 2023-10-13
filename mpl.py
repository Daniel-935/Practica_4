import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#!Creamos la clase base para la red neuronal que toma como clase padre la clase "nn" de pytorch
class NeuralNet(nn.Module):

    #*El constructor toma el numero de inputs, capas, neuronas y salias y el nombre del archivo para tomar las entradas y salidas
    def __init__(self, noInputs, noCapas, noNeuronas, noOutput):
        #*Llama a la clase padre para hacer nuestro modelo de la red neuronal
        super(NeuralNet, self).__init__()

        #*Crea la primer capa de entrada y de salida
        self.capaInput = nn.Linear(noInputs, noNeuronas)
        self.capaOutput = nn.Linear(noNeuronas, noOutput)

        #*Crea las capas ocultas y las almacena en una lista de torch
        #*Cada capa del tipo lineal
        self.capasOcultas = nn.ModuleList()
        for i in range(noCapas):
            self.capasOcultas.append(nn.Linear(noNeuronas, noNeuronas))

    #*Metodo para hacer el feedforward
    def forward(self, inputs):
        #*Comienza por recorrer todas las capas ocultas y usar la funcion de activacion con el fin de obtener una prediccion
        #*Inicia con la capa de entrada y recorre la lista creada de torch, se va a usar la funcion sigmoide
        prediction = torch.sigmoid(self.capaInput(inputs))
        for capa in self.capasOcultas:
            prediction = torch.sigmoid(capa(prediction))
        #*Obtiene la prediccion final (capa de salida)
        prediction = torch.sigmoid(self.capaOutput(prediction))
        return prediction