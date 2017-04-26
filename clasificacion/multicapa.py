#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Para silenciar unos Warnings fastidiosos	
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from random import randint
import math


class Detector:

	def recortar(self):
		start_time = time.time()


		img = cv2.imread('placa1.jpg',0)
		#img = cv2.imread('texto1.JPG',0)


		# Se hace un filtro Gaussiano 
		blur = cv2.GaussianBlur(img,(5,5),0)

		# Se binariza la imagen con el algoritmo de OTSU 
		_,binaria = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

		# Se hace una cerradura para eliminar un poco de basura 
		kernel = np.ones((3,3),np.uint8)
		imagenBiaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)
		imagenBiaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)
		#cv2.imshow("Binaria", imagenBiaria)

		# Se generan los borde canny a partir de la imagen binarizada
		bordes = cv2.Canny(binaria, 30, 200)

		# Se encuentran los contornos a partir de los bordes canny
		#_, contours, hierarchy = cv2.findContours(bordes,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contours, hierarchy = cv2.findContours(bordes,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		# Calcula el area de la imagen que se esta procesando
		areaImagen = len(img) * len(img[0]) 

		nuevosContornos = []

		# Se filtran los contornos que tengan un area menor al 2% del area de la imagen
		i = 0
		for contorno in contours:
			if cv2.contourArea(contorno) >= (areaImagen * 2) / float(100) and i %2 == 0 and cv2.contourArea(contorno) <= (areaImagen * 50) / float(100):
			#if cv2.contourArea(contorno) >= (areaImagen * 2) / float(100):
				
				nuevosContornos.append(contorno) # Agrega el contorno
				
				# Se muestra el contorno y el rectangulo en la imgane binarizada
				cv2.drawContours(binaria, contours, i, (120,243,0), 3)# Dibuja el contorno
				x,y,w,h = cv2.boundingRect(contorno) # Encuentra el rectangulo que se ajusta al contorno
				cv2.rectangle(binaria,(x,y),(x+w,y+h),(0,255,0),2) # Dibuja el rectangulo		

				#recorte = img[y:y+h, x:x+w] # Recorta el pedazo de itenres de la imagen
				
				#if len(recorte) > 0: # El recorte coniene algo
					#print str(len(recorte)) + " " + str(len(recorte[0]))
				#	cv2.imshow("recorte" + str(i), recorte)
			i += 1

		#cv2.imshow("binaria", binaria) # Muestra la imagen binarizada

		contornosOrdenados = nuevosContornos[:len(nuevosContornos)] # copia de los contornos identificados

		# se usa el metodo de ordenamiento burbuja
		i = 0
		size = len(contornosOrdenados)

		while (i < size):
		        j=i   
		        while(j < size):
		                x0,y0,w0,h0 = cv2.boundingRect(contornosOrdenados[i]) # encuentra el recuadro para i
		                x1,y1,w1,h1 = cv2.boundingRect(contornosOrdenados[j]) # encuentra el recuadro para j
		                if(x0 > x1): # compara para saber quien es mayor
		                        # cambio por temporal
		                        temp = contornosOrdenados[i]
		                        contornosOrdenados[i] = contornosOrdenados[j]
		                        contornosOrdenados[j] = temp
		        
		                j=j+1

		        i=i+1 
		                

		# Se visualiza los recortes ordenados
		i = 0
		for contorno in contornosOrdenados:
		        x,y,w,h = cv2.boundingRect(contorno) # encuentra el recuadro
		        recorte = img[y:y+h, x:x+w] # recorte de la imagen
		        if len(recorte) > 0: # El recorte coniene algo
					#print str(len(recorte)) + " " + str(len(recorte[0]))
					a = 1
					#cv2.imshow("recorte" + str(i), recorte)	# se muetra
		        i = i+1

		#cv2.imshow("recorte" + str(), recorte) # se muestra

		# transformacion a 28 *28

		cv2.normalize(img,  img, 0, 1, cv2.NORM_MINMAX)
		#print img


		width = 28
		height = 28

		i = 0
		recortes = []
		for contorno in contornosOrdenados:
		        x,y,w,h = cv2.boundingRect(contorno) # encuentra el recuadro
		        recorte = img[y:y+h, x:x+w] # recorte de la imagen
		        if len(recorte) > 0: # El recorte coniene algo
		                        #print str(len(recorte)) + " " + str(len(recorte[0]))		                   
		                recorte = cv2.resize(recorte, (width,height)) # redimension de 28 * 28
		                #recorte = 1 / recorte
		                recortes.append(recorte)
		                #cv2.imwrite("/uptc/Inteligencia Computacional/deteccion_texto/img" + str(i) + ".jpg",recorte)
		                #cv2.imshow("recorte2" + str(i), recorte)	# se muetra
		        i = i+1


		#cv2.imshow("bordes", binaria)

		# se muestra el tiempo de ejecucion
		print("--- %s seconds ---" % (time.time() - start_time))
		

		## Salir con ESC
		#while(1):
			#a = 1
		#    # Mostrar la mascara final y la imagen
		#    #cv2.imshow("Origial", img)
		#    tecla = cv2.waitKey(5) & 0xFF
		#    if tecla == 27:
#		        break
#
#		cv2.destroyAllWindows()
		return recortes


#*********************************************************** Clasificador*******************++++++

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

n_hidden = 250 # cantidad de neuronas capa oculta

x = tf.placeholder(tf.float32, [None, 784]) # Entradas 
y_ = tf.placeholder(tf.float32, [None, 10]) # Salidas ideales

# Pesos y vias(no se sabe)
W0 = tf.Variable(tf.random_normal([784, n_hidden], stddev=0.01)) # Pesos psiapticos
B0 = tf.Variable(tf.random_normal([n_hidden], stddev=0.01))

W1 = tf.Variable(tf.random_normal([n_hidden, 10], stddev=0.01)) # Pesos psiapticos
B1 = tf.Variable(tf.random_normal([10], stddev=0.01))

h = tf.nn.tanh(tf.matmul(x, W0) + B0) # Salida capa oculta
y = tf.nn.tanh(tf.matmul(h, W1) + B1) # Salida

mse = tf.reduce_mean(tf.square(y - y_)) # Funcion de error "min sqare error"

#Minimizar el error
train_step = tf.train.AdamOptimizer(0.005).minimize(mse)

# Inicializar las variables
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

trainLoss = [] # Datos de entrenamiento en cada paso
testLoss = [] # Evaluar el entrenamiento en cada paso

# Entranado la red -- no sobre todo el conjunto de etrenamiento sino de una muestra estocastica-- 150 iteraciones
for i in range(1,80):
	batchx, batchy = mnist.train.next_batch(500) # Muestra estocastica de 1000 imagenes
	print batchy[0]
	# Ejecuta una sesion de entrenamiento
	sess.run(train_step, feed_dict = {x:batchx, y_:batchy} ) # feed_dict es un diccionario con los datos 	

	loss1 = sess.run(mse, feed_dict = {x:batchx, y_:batchy} ) # Error sobre los datos de entrenamiento
	loss2 = sess.run(mse, feed_dict = {x:mnist.test.images, y_:mnist.test.labels} ) # Error sobre los datos de evaluacion

	trainLoss.append(loss1)
	testLoss.append(loss2)
	pass
print "Entrenamiento terminado..."

prediction = sess.run(y, feed_dict={x:mnist.test.images ,y_: mnist.test.labels})

errors = 0.0
for i in range(prediction.shape[0]):
	if np.argmax(prediction[i]) != np.argmax(mnist.test.labels[i]):
		errors += 1

print "Test error : " + str(errors/(prediction.shape[0]))



detector = Detector()
recortes = detector.recortar()
recortes[3].reshape(784)


#prueba = sess.run(y, feed_dict={x:[mnist.test.images[3]]})
prueba = sess.run(y, feed_dict={x:[recortes[4].reshape(784)]})
print "Resultado index clasificación : " + str(np.argmax(prueba)) # Muestra el indice del digito con mas probabilidad
#Muestra la imagen que se esta clasiicando
#plt.imshow(mnist.test.images[3,:].reshape(28,28), cmap="gray")
plt.imshow(recortes[4])
plt.show()