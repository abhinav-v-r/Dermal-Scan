class_labels = ['clear_faces', 'dark_spots', 'puffy_eyes', 'wrinkles']
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2

def predict_skin_condition(img_path, model, class_labels):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize

    # Prediction
    predictions = model.predict(img_array)[0]
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Display
    plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%)")
    plt.show()

    return predicted_class, predictions


import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Paths
faceProto = "/Dermal_Scan/models/age_prediction/opencv_face_detector.pbtxt"
faceModel = "/Dermal_Scan/models/age_prediction/opencv_face_detector_uint8.pb"
ageProto = "/Dermal_Scan/models/age_prediction/age_deploy.prototxt"
ageModel = "/Dermal_Scan/models/age_prediction/age_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# Original ageList
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
class_labels = ['clear_faces', 'dark_spots', 'puffy_eyes', 'wrinkles'] # Define class_labels here

# Load models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)

def predict_age(image_path):
    frame = cv2.imread(image_path)
    # Resize the image for better face detection on some images
    frame = cv2.resize(frame, (600, 600))
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)

    faceNet.setInput(blob)
    detections = faceNet.forward()

    h, w = frame.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # Increased confidence threshold for better accuracy
        if confidence > 0.8:
            x1, y1, x2, y2 = int(detections[0, 0, i, 3]*w), int(detections[0, 0, i, 4]*h), \
                             int(detections[0, 0, i, 5]*w), int(detections[0, 0, i, 6]*h)
            face = frame[y1:y2, x1:x2]

            if face.shape[0] > 0 and face.shape[1] > 0: # Check if the face region is not empty
                blob2 = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                ageNet.setInput(blob2)
                agePreds = ageNet.forward()
                # Use the index with the highest probability to get the predicted age range
                age = ageList[agePreds[0].argmax()]
                return age
    return "No face detected"

# Define predict_skin_condition function here to use in this cell
def predict_skin_condition(img_path, model, class_labels, age):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize

    # Prediction
    predictions = model.predict(img_array)[0]
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Display
    plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%) Age: {age}") # Include age in title
    plt.show()

    return predicted_class, predictions

test_img = "/content/drive/MyDrive/-Age-Detection-master/kid2.jpg"
age = predict_age(test_img) # Calculate age once
pred_class, probs = predict_skin_condition(test_img, model, class_labels, age) # Pass age to the function

print("Class probabilities:", dict(zip(class_labels, probs)))