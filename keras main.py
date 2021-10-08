from keras.models import load_model
from time import sleep
from keras.preprocessing.image 
from keras.preprocessing import image
import cv2
import numpy to np

face_classifier = cv2.CascadeClassifier(r'E:\Programming\facial recog\haarcascade_frontalface_default.xml')
classifier =load_model(r'E:\Programming\facial recog\model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)



while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h))
        roi_gray = [y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray]):
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)
            label=emotion_labels[prediction]
            label_position 
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
