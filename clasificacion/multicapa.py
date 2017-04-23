# Para silenciar unos Warnings fastidiosos	
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

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

pesos = sess.run(W0)

for i in xrange(0,10):
	plt.imshow(pesos[:, i].reshape(28,28), cmap="bwr")
	plt.show()
	pass

#plt.subplot(211)
plt.plot(trainLoss, 'r')
#plt.title("train")

#plt.subplot(212)
plt.plot(testLoss, 'b')
#plt.title("test")

plt.show()