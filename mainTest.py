import cv2
from keras.models import load_model
import numpy as np

model=load_model('./training/model-017.model')
video=cv2.VideoCapture(0)
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

color_dict={0:(0,0,255),1:(0,255,0)}
labels_dict={0:"Felame",1: "Male"}

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray, 1.3, 3)
    for x,y,w,h in faces:
        sub_face_img=gray[y:y+h,x:x+h]
        resized=cv2.resize(sub_face_img,(32,32))
        normalize=resized/255.0
        reshape=np.reshape(normalize,(1,32,32,1))
        result=model.predict(reshape)
        label=np.argmax(result,axis=1)[0]
        print(label)
        cv2.rectangle(frame,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()
