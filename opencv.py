import cv2 as cv
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

model = tf.keras.models.load_model("Model/Traffic_Sign_Classifier_CNN.hdf5")

ip = "http://192.168.0.178:8080/video"
capture = cv.VideoCapture(ip)

signNames = {
        0: 'Speed limit (20km/h)',
        1: 'Speed limit (30km/h)',
        2: 'Speed limit (50km/h)',
        3: 'Speed limit (60km/h)',
        4: 'Speed limit (70km/h)',
        5: 'Speed limit (80km/h)',
        6: 'End of speed limit (80km/h)',
        7: 'Speed limit (100km/h)',
        8: 'Speed limit (120km/h)',
        9: 'No passing',
        10: 'No passing for vehicles over 3.5 metric tons',
        11: 'Right-of-way at the next intersection',
        12: 'Priority road',
        13: 'Yield',
        14: 'Stop',
        15: 'No vehicles',
        16: 'Vehicles over 3.5 metric tons prohibited',
        17: 'No entry',
        18: 'General caution',
        19: 'Dangerous curve to the left',
        20: 'Dangerous curve to the right',
        21: 'Double curve',
        22: 'Bumpy road',
        23: 'Slippery road',
        24: 'Road narrows on the right',
        25: 'Road work',
        26: 'Traffic signals',
        27: 'Pedestrians',
        28: 'Children crossing',
        29: 'Bicycles crossing',
        30: 'Beware of ice/snow',
        31: 'Wild animals crossing',
        32: 'End of all speed and passing limits',
        33: 'Turn right ahead',
        34: 'Turn left ahead',
        35: 'Ahead only',
        36: 'Go straight or right',
        37: 'Go straight or left',
        38: 'Keep right',
        39: 'Keep left',
        40: 'Roundabout mandatory',
        41: 'End of no passing',
        42: 'End of no passing by vehicles over 3.5 metric tons'}

while True:
    isTrue, frame = capture.read()
    img = cv.resize(frame, (32, 32))  # Resize the frame
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert to grayscale
    img = img / 255.0
    image = np.reshape(img, [1, 32, 32, 1])  # Grayscale image has a single channel

    classes = model.predict(image)
    predict = np.argmax(classes)
    predictClass = signNames[predict]
    frame = cv.resize(frame, (400, 400))
    cv.putText(frame, predictClass, (40, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv.imshow("Video", frame)

    if(cv.waitKey(12) & 0xFF==ord('d')):
        break

capture.release()
capture.destroyAllWindows()