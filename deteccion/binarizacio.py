import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from random import randint
import math
from PIL import Image, ImageDraw

start_time = time.time()


img = cv2.imread('placa1.jpg',0)


# Se hace un filtro Gaussiano 
blur = cv2.GaussianBlur(img,(5,5),0)

# Se binariza la imagen con el algoritmo de OTSU 
_,binaria = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Se hace una cerradura
kernel = np.ones((3,3),np.uint8)
imagenBiaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)

cv2.imshow("Binaria", imagenBiaria)

bordes = cv2.Canny(binaria, 30, 200)
cv2.imshow("canny", bordes)

im2, contours, hierarchy = cv2.findContours(binaria,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	#contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0,120,0), 3)
cv2.imshow("bordes", img)


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

