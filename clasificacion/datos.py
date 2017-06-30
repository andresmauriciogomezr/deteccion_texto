#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import json
import base64
import numpy as np
import time

start_time = time.time()

width = 28
height = 28

archivo = open("listaImagenes.txt", "r")

images = []
labels = []
indexLabel = 0

cantidadEjemplos = 10160

def nextPaquete(cantidadBatch):
	aleatorios = np.random.uniform(0,cantidadEjemplos,cantidadBatch)
	batchy = []
	batchx = []

	for x in xrange(0,len(aleatorios)):
		#print (int)(aleatorios[x])
		batchy.append(labels[(int)(aleatorios[x])])
		batchx.append(images[(int)(aleatorios[x])])
		pass

	return batchx, batchy


i = 0
for linea in archivo.readlines():
	ruta = str('Fnt/'+linea[:-1]+'.png')	
	
	img = cv2.imread(ruta,0)
	
	cv2.normalize(img,  img, 0, 1, cv2.NORM_MINMAX)
	
	img = cv2.resize(img, (width,height)) # redimension de 28 * 28	

	if i == 1016: # cada Caracter tiene 1016 ejemploss
		indexLabel += 1
		i = 0
	
	#if i < 1016/2: # Solo agrega la mitad
	images.append(img.reshape(784))

	label = np.zeros(10)
	label[indexLabel] = 1
	labels.append(label)

	i+=1

print("--- %s seconds ---" % (time.time() - start_time))
print indexLabel
print "-- Imagenes de entrenamiento cargadas..."

entrenamiento = {'images' : images, 'labels' : labels}

"""
x, y = nextPaquete(1000)

#print len(x)
#print len(y)

i = 36

print y[i]
print np.argmax(y[i])



plt.imshow(x[i].reshape(28,28))
plt.show()
"""
