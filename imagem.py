# pybrain
from pybrain.datasets.supervised import SupervisedDataSet 
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

# processamento de imagens
from PIL import Image

# gráficos 
import matplotlib.pyplot as plt
import numpy as np
import os

os.remove("output.txt")
os.remove("resultado.txt")

file = open("output.txt","w")
result = open("resultado.txt","w")

# função para carregar os dados de treinameto a partir das imagens
def getDataImage( path):
    #Read image
    img = Image.open( path )
    img = img.resize( (40,40) )
    pixels = img.load() 
    data = []
    pixel = []
    for i in range( img.size[0]):         # for every col do: img.size[0]
        for j in range( img.size[1] ):    # for every row   img.size[1]      
            pixel = pixels[i,j]          # get every pixel
            data.append( pixel[0] )
            data.append( pixel[1] )
            data.append( pixel[2] )

    #Viewing EXIF data embedded in image
    # exif_data = img._getexif()
    # exif_data
    return data

# carregando a primeira imagem
dataTraining =  getDataImage( 'img\\0a.jpg')
size = 40 * 40 * 3

# configurando a rede neural artificial e o dataSet de treinamento
network = buildNetwork( size, 100, 30, 4 )  # define network
dataSet = SupervisedDataSet( size, 4 )      # define dataSet


# load dataSet
dataSet.addSample ( getDataImage('img\\0a.jpg'), (0, 0, 0, 0) )  # Não lubrificado
dataSet.addSample ( getDataImage('img\\0b.jpg'), (0, 0, 0, 0) )  # Não lubrificado
dataSet.addSample ( getDataImage('img\\0c.jpg'), (0, 0, 0, 0) )  # Não lubrificado
dataSet.addSample ( getDataImage('img\\0d.jpg'), (0, 0, 0, 0) )  # Não lubrificado
dataSet.addSample ( getDataImage('img\\0e.jpg'), (0, 0, 0, 0) )  # Não lubrificado
dataSet.addSample ( getDataImage('img\\0f.jpg'), (0, 0, 0, 0) )  # Não lubrificado

dataSet.addSample ( getDataImage('img\\1a.jpg'), (1, 1, 1, 1) ) # Lubrificado
dataSet.addSample ( getDataImage('img\\1b.jpg'), (1, 1, 1, 1) ) # Lubrificado
dataSet.addSample ( getDataImage('img\\1c.jpg'), (1, 1, 1, 1) ) # Lubrificado
dataSet.addSample ( getDataImage('img\\1d.jpg'), (1, 1, 1, 1) ) # Lubrificado
dataSet.addSample ( getDataImage('img\\1e.jpg'), (1, 1, 1, 1) ) # Lubrificado
dataSet.addSample ( getDataImage('img\\1f.jpg'), (1, 1, 1, 1) ) # Lubrificado


# trainer
trainer = BackpropTrainer( network, dataSet)
error = 1
iteration = 0
outputs = []
while error > 0.001: 
    error = trainer.train()
    outputs.append( error )
    iteration += 1    
    print ( iteration, error )
    file.write(str(error) + "\n")

# plot graph
plt.ioff()
plt.plot( outputs )
plt.xlabel('Iterações')
plt.ylabel('Erro Quadrático')
plt.show()

# Fase de teste
name = ['lub_1.jpg', 'lub_2.jpg', 'lub_3.jpg', 'n_lub_1.jpg', 'n_lub_2.jpg', 'n_lub_3.jpg']
for i in range( len(name) ):
    path = "img\\test\\" + name[i]
    print ( path )
    resultTest = network.activate( getDataImage( path ) )
    print (resultTest)

    result.write(str(path) + "\n")
    result.write(str(resultTest) + "\n\n")