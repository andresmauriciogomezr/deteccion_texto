import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from random import randint
import math


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
im2, contours, hierarchy = cv2.findContours(bordes,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

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

cv2.imshow("binaria", binaria) # Muestra la imagen binarizada

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
                cv2.imshow("recorte" + str(i), recorte)	# se muetra
        i = i+1


# transformacion a 28 *28

width = 28
height = 28

i = 0
for contorno in contornosOrdenados:
        x,y,w,h = cv2.boundingRect(contorno) # encuentra el recuadro
        recorte = img[y:y+h, x:x+w] # recorte de la imagen
        if len(recorte) > 0: # El recorte coniene algo
                        #print str(len(recorte)) + " " + str(len(recorte[0]))
                recorte = cv2.resize(recorte, (width,height)) # redimension de 28 * 28
                cv2.imshow("recorte" + str(i), recorte)	# se muetra
        i = i+1


#cv2.imshow("bordes", binaria)

# se muestra el tiempo de ejecucion
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()

# Salir con ESC
while(1):
    # Mostrar la mascara final y la imagen
    cv2.imshow("Origial", img)
    tecla = cv2.waitKey(5) & 0xFF
    if tecla == 27:
        break

cv2.destroyAllWindows()

