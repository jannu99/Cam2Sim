#!/usr/bin/env python3

#import tensorflow as tf
#from tensorflow import keras
from keras.models import load_model

import numpy as np
import cv2
from PIL import Image
import time
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


FIXED_THROTTLE = False
MAX_STEERING = 9.0
STEERING = 0
THROTTLE = 1

class Dave2Model:

    model = None

    def parse_model_outputs(self,outputs):
            res = []
            for i in range(outputs.shape[1]):
                res.append(outputs[0][i])

            return res
    
    def __init__(self,model_path):
         #model_path = os.path.join(model_name)
        self.model = load_model(model_path, compile=False, safe_mode=False)
        self.model.compile(loss="sgd", metrics=["mse"])

    def calculate_dave2_image(self,image):
        # Resize with PIL (returns a new image)
        image = image.resize((200, 66))
        #image.save("prev.png")

        image_array = np.array(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Step 2: add batch dimension
        # Model expects shape: (1, H, W, C)
        image_array = image_array[None, ...]

        # Step 3: (optional) check shape
        #print("Model input shape:", image_array.shape)

        outputs = self.model.predict(image_array, verbose=0)
        parsed_outputs = self.parse_model_outputs(outputs)

        steering = 0.
        throttle = 0.
        if len(parsed_outputs) > 0:        
            steering = parsed_outputs[STEERING] #*1.4
            throttle = parsed_outputs[THROTTLE]

        if FIXED_THROTTLE:
            throttle = 1.

        steering = min(steering, MAX_STEERING)
        steering = max(steering, MAX_STEERING * -1.0)

        return steering, throttle