from mimetypes import init
from pyexpat import model
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np 
import cv2 as cv
import imutils

class GradCAM:
    
    def __init__(self, model, classIdx, layerName=None) -> None:
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName

        if self.layerName is None:
            self.layerName = self.find_target_layer()
        
    def find_target_layer(self):
        """Designed to find final conv+pooling layer in the CNN to produce class activation map"""
        for layer in reversed(self.model.layers):
            if len(layer.output_shape)==4:
                return layer.name
            else: 
                raise ValueError("Could not find layer with 4D output")

    def computeHeatmap(self, image, eps=1e-8):
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
            self.model.output]
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            convOutputs, predictions = gradModel(inputs)
            loss = predictions[L, self.classIdx]

        grads = tape.gradient(loss, convOutputs)