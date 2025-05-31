
#!/usr/bin/python3

# Capture a full resolution image to memory rather than to a file.

import time

from picamera2 import Picamera2, Preview

picam2 = Picamera2()
picam2.start_preview(Preview.QTGL)
preview_config = picam2.create_preview_configuration()
capture_config = picam2.create_still_configuration()

picam2.configure(preview_config)
picam2.start()
time.sleep(2)

image = picam2.switch_mode_and_capture_image(capture_config)
image.show()


time.sleep(5)

picam2.close()


# from keras.models import load_model
# from time import sleep
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing import image
# import cv2
# import numpy as np

# face_classifier = cv2.CascadeClassifier(
#     'haarcascade_frontalface_default.xml')
# # emotion_model = load_model('emotion_detection_model_100epochs.h5')


# class_labels = ['Angry', 'Disgust', 'Fear',
#                 'Happy', 'Neutral', 'Sad', 'Surprise']


# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     labels = []

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

#         # Get image ready for prediction
#         roi = roi_gray.astype('float')/255.0  # Scale
#         roi = img_to_array(roi)
#         # Expand dims to get it ready for prediction (1, 48, 48, 1)
#         roi = np.expand_dims(roi, axis=0)

#         # Yields one hot encoded result for 7 classes
#         # preds = emotion_model.predict(roi)[0]
#         # label = class_labels[preds.argmax()]  # Find the label
#         # label_position = (x, y)
#         # cv2.putText(frame, label, label_position,
#         #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     cv2.imshow('Emotion Detector', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
