import cv2
import numpy as np
from matplotlib import pyplot as plt
import time


def meanShift():

	#img = cv2.imread('deteccion/figuras.png')
	img = cv2.imread('placa1.jpg')

	img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

	#cv2.imshow("sisas", img)


	resultado = cv2.pyrMeanShiftFiltering(img, 25, 40, 3)
	#cv2.imshow("primera", resultado)

	cantidadFiltros = 0
	for x in xrange(0,cantidadFiltros):
		resultado = cv2.pyrMeanShiftFiltering(resultado, 25, 35, 3)
		pass

	#cv2.imshow("segunda", resultado)

	kernel = np.ones((3,3),np.uint8)
	resultado = cv2.morphologyEx(resultado, cv2.MORPH_CLOSE, kernel)
	resultado = cv2.morphologyEx(resultado, cv2.MORPH_OPEN, kernel)

	imgray = cv2.cvtColor(resultado,cv2.COLOR_BGR2GRAY)
	bordes = cv2.Canny(imgray, 30, 200)
	
	ret,thresh = cv2.threshold(imgray,127,255,0)
	
	im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	#contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	cv2.drawContours(resultado, contours, -1, (0,255,0), 3)


	
	plt.subplot(411)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#plt.hist(img.ravel(),256,[0,256])


	plt.subplot(412)
	#plt.hist(img.ravel(),256,[0,256])

	plt.subplot(413)
	#plt.hist(gray_image.ravel(),256,[0,256])
	#plt.hist(resultado.ravel(),256,[0,256])

	plt.subplot(414)
	plt.hist(imgray.ravel(),256,[0,256])
	print("--- %s seconds ---" % (time.time() - start_time))

	#plt.show()

		# Salir con ESC
	while(1):
	    # Mostrar la mascara final y la imagen
	    cv2.imshow("Ultima", resultado)
	    tecla = cv2.waitKey(5) & 0xFF
	    if tecla == 27:
	        break

	cv2.destroyAllWindows()


start_time = time.time()
meanShift()

#nuevo()