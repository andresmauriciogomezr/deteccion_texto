#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Para silenciar unos Warnings fastidiosos	
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784]) # Entradas 
y_ = tf.placeholder(tf.float32, [None, 10]) # Salidas ideales

W = tf.Variable(tf.zeros([784, 10])) # Pesos psiapticos
B = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + B)
#y = tf.nn.tanh(tf.matmul(x, W) + B) # Salida

mse = tf.reduce_mean(tf.square(y - y_)) # Funcion de error "min sqare error"

#Minimizar el error
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(mse)

# Inicializar las variables
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

trainLoss = [] # Datos de entrenamiento en cada paso
testLoss = [] # Evaluar el entrenamiento en cada paso

# Entranado la red -- no sobre todo el conjunto de etrenamiento sino de una muestra estocastica-- 150 iteraciones
for i in range(1,150):
	batchx, batchy = mnist.train.next_batch(1000) # Muestra estocastica de 1000 imagenes

	# Ejecuta una sesion de entrenamiento
	sess.run(train_step, feed_dict = {x:batchx, y_:batchy} ) # feed_dict es un diccionario con los datos 	

	loss1 = sess.run(mse, feed_dict = {x:batchx, y_:batchy} ) # Error sobre los datos de entrenamiento
	loss2 = sess.run(mse, feed_dict = {x:mnist.test.images, y_:mnist.test.labels} ) # Error sobre los datos de evaluacion

	trainLoss.append(loss1)
	testLoss.append(loss2)
	pass
print "Entrenamiento terminado..."

# Calculando la probabilidad error
prediction = sess.run(y, feed_dict={x:mnist.test.images ,y_: mnist.test.labels})

errors = 0.0
for i in range(prediction.shape[0]):
	if np.argmax(prediction[i]) != np.argmax(mnist.test.labels[i]):
		errors += 1

print "Probabilidad error : " + str(errors/(prediction.shape[0]))

""" muestra los pesos --- no borrar -- descomentar para ver los pesos
pesos = sess.run(W)

for i in xrange(0,10):
	plt.imshow(pesos[:, i].reshape(28,28), cmap="bwr")
	plt.show()
	pass
"""


# *********** Prueba *******************

# Se hace la clasificación de la imagen numero dos de las imagenes de prueba
prediction = sess.run(y, feed_dict={x:[mnist.test.images[2]]}) 
print "Resultado index clasificación : " + str(np.argmax(prediction)) # Muestra el indice del digito con mas probabilidad

#Muestra la imagen que se esta clasiicando
plt.imshow(mnist.test.images[2,:].reshape(28,28), cmap="gray")
plt.show()


#print mnist.test.images[3:]


plt.plot(trainLoss, 'r')
plt.plot(testLoss, 'b')

#plt.show()
