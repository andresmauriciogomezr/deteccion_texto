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
i = 0
for linea in archivo.readlines():
	ruta = str('Fnt/'+linea[:-1]+'.png')	
	
	img = cv2.imread(ruta,0)
	
	cv2.normalize(img,  img, 0, 1, cv2.NORM_MINMAX)
	
	img = cv2.resize(img, (width,height)) # redimension de 28 * 28

	images.append(img)

	if i == 1016:
		indexLabel += 1
		i = 0

	label = np.zeros(62)
	label[indexLabel] = 1
	labels.append(label)
	i+=1

print 'ya'
print i
print indexLabel
print("--- %s seconds ---" % (time.time() - start_time))

print np.argmax(labels[40000])

plt.imshow(images[40000])
plt.show()