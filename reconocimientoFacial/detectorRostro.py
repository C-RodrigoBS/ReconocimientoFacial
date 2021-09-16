import cv2
import numpy as np
import os
import imutils

personaCreada="Rodrigo"
dataPath= "E:/OmesTutorials-master/Detección de Rostros/data"
personaPath= dataPath + "/" + personaCreada

if not os.path.exists(personaPath):
    print(f"Carpeta creada: {personaPath}")
    os.makedirs(personaPath)

cap = cv2.VideoCapture(0)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
contador=0

#se inicia la lectura de cada fotograma del video
while True:
	ret,frame = cap.read()
    if ret==False: break
    frame = imutils.resize(frame, width=640)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

	faces = faceClassif.detectMultiScale(gray, 1.1, 5)
    

	#recorre la información optenida de los rostros
	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        rostro = auxFrame[x:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personaPath + "/rostro_{}.jpg".format(count),rostro)
        contador=+1
	cv2.imshow('frame',frame)
	
    key= cv2.waitKey(1)
    if key == 27 or count >=300:
        break

	#if cv2.waitKey(1) & 0xFF == ord('q'):
	#	break
cap.release()
cv2.destroyAllWindows()
