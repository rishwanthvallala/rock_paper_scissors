import cv2
import numpy as np
import math
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import os

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress TensorFlow progress bar
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

class CameraInterface:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=1)
        self.imgSize = 300
        self.offset = 20
        
        try:
            with tf.keras.utils.custom_object_scope({'DepthwiseConv2D': CustomDepthwiseConv2D}):
                self.model = tf.keras.models.load_model("Model/keras_model.h5", compile=False)
        except Exception as e:
            print(f"Failed to load Keras model: {str(e)}")
            print("Trying TFLite conversion...")
            model = tf.keras.models.load_model("Model/keras_model.h5", custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}, compile=False)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            with open('model.tflite', 'wb') as f:
                f.write(tflite_model)
            self.interpreter = tf.lite.Interpreter(model_content=tflite_model)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        
        with open("Model/labels.txt", "r") as file:
            self.labels = [line.strip() for line in file.readlines()]

    def get_frame(self):
        success, img = self.cap.read()
        if not success:
            return None, None
        
        imgOutput = img.copy()
        hands, img = self.detector.findHands(img)
        
        detected_move = None
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
            imgCrop = img[max(0, y - self.offset):min(img.shape[0], y + h + self.offset),
                          max(0, x - self.offset):min(img.shape[1], x + w + self.offset)]

            aspectRatio = h / w

            if aspectRatio > 1:
                k = self.imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
                wGap = math.ceil((self.imgSize - wCal) / 2)
                imgWhite[:, wGap:wGap+wCal] = imgResize
            else:
                k = self.imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
                hGap = math.ceil((self.imgSize - hCal) / 2)
                imgWhite[hGap:hGap+hCal, :] = imgResize

            prediction, index = self.get_prediction(imgWhite)

            detected_move = self.labels[index]
            cv2.rectangle(imgOutput, (x - self.offset, y - self.offset-50),
                          (x - self.offset+90, y - self.offset-50+50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, detected_move, (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x-self.offset, y-self.offset),
                          (x + w+self.offset, y + h+self.offset), (255, 0, 255), 4)

        return imgOutput, detected_move

    def get_prediction(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0).astype(np.float32)

        if hasattr(self, 'interpreter'):
            self.interpreter.set_tensor(self.input_details[0]['index'], img)
            self.interpreter.invoke()
            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
        else:
            prediction = self.model.predict(img)
        
        index = np.argmax(prediction)
        return prediction[0], index

    def release(self):
        self.cap.release()