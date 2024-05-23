# Importing the libraries
from PIL import Image
import cv2
from keras.models import load_model
import numpy as np
from collections import Counter

# Initialize variables
sample_count = 15
predictions = []

# Function to determine the most common class
def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

from keras.preprocessing import image
model = load_model(r"D:\Testing\facefeatures_new_modelAddRegMoreDataSGDLowTR40MoreEpochsHigherLR.h5")

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

while sample_count > 0:
    _, frame = video_capture.read()
    face = face_extractor(frame)
    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        im = Image.fromarray(face, 'RGB')
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        name = "Unknown"
        
        # Modify these conditions to force a class name
        if pred[0][0] > 0.3:
            name = 'Sahil'
        elif pred[0][1] > 0.3:
            name = 'Manamrit'
        elif pred[0][2] > 0.3:
            name = 'Himmat'
        elif pred[0][3] > 0.3:
            name = 'Abhijot'

        predictions.append(name)
        sample_count -= 1

        # Display the current detected name on the video frame
        cv2.putText(frame, "Detecting...", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        #cv2.putText(frame, f"Class: {name}", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Determine the most common class from predictions
most_common_class = most_common(predictions)

# Display the most common class on the video window
cv2.putText(frame, f"Hello {most_common_class}", (50, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
cv2.imshow('Video', frame)

# Wait for a key press (you can customize this behavior)
cv2.waitKey(0)

video_capture.release()
cv2.destroyAllWindows()

       

