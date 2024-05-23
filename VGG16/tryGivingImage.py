# Importing the libraries
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
import numpy as np
import cv2

# Load the pre-trained VGG16 model
model = load_model(r'D:\vgg16\code\Deep-Learning-Face-Recognition-master\facefeatures_new_modelAddRegMoreData1.h5')

# Loading the cascades
face_cascade = cv2.CascadeClassifier(r'D:\vgg16\code\Deep-Learning-Face-Recognition-master\haarcascade_frontalface_default.xml')


# Function to recognize the face in the input image
def recognize_face(image_path):
    # Load the input image
    img = cv2.imread(image_path)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if len(faces) == 0:
        return "No face found", None
    
    # Process the first detected face (assuming there's only one)
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (224, 224))
    
    # Preprocess the face for model prediction
    face = preprocess_input(face)
    
    # Use the model to predict the name and confidence
    prediction = model.predict(np.array([face]))
    
    names = ['Abhijot', 'Himmat', 'Manamrit', 'Sahil']
    
    # Check which name has the highest confidence
    max_confidence_index = np.argmax(prediction)
    name = names[max_confidence_index]
    
    return name, prediction

# Input the image path
#image_path = r"D:\vgg16\code\Deep-Learning-Face-Recognition-master\dataset\Train\Sahil\89.jpg"
image_path = r"D:\vgg16\code\Deep-Learning-Face-Recognition-master\dataset\Validation\Himmat\97.jpg"
name, confidence = recognize_face(image_path)

print("Predicted Name:", name)
print("Confidence Values:", confidence)
