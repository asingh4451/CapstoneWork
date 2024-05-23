# Importing the libraries
from PIL import Image
import cv2
from keras.models import load_model
import numpy as np

from keras.preprocessing import image
model = load_model(r'D:\vgg16\code\Deep-Learning-Face-Recognition-master\facefeatures_new_modelAddRegMoreDataSGD.h5')

# Loading the cascades
face_cascade = cv2.CascadeClassifier(r'D:\vgg16\code\Deep-Learning-Face-Recognition-master\haarcascade_frontalface_default.xml')

def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    face = face_extractor(frame)
    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        im = Image.fromarray(face, 'RGB')
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        print(pred)
                     
        name = "Unknown"  # Default name is "Unknown"

        # Modify these conditions to force a class name
        if pred[0][0] > 0.5:
            name = 'Sahil'
        elif pred[0][1] > 0.5:
            name = 'Manamrit'
        elif pred[0][2] > 0.5:
            name = 'Himmat'
        elif pred[0][3] > 0.5:
            name = 'Abhijot'
        frame = cv2.rectangle(frame, (0, 0), (300, 60), (0, 0, 0), -1)
        cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()

       

