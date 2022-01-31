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
     
        raise ValueError("Could not find layer with 4D output")

    def computeHeatmap(self, image, eps=1e-8):
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
            self.model.output]
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        grads = tape.gradient(loss, convOutputs)
        castConvOutputs = tf.cast(convOutputs>0, 'float32')
        castGrads = tf.cast(grads>0, 'float32')
        guidedGrads = castConvOutputs * castGrads * grads
        guidedGrads = guidedGrads[0]
        convOutputs = convOutputs[0]

        # Compute average of gradient values to use as weights
        weights = tf.reduce_mean(guidedGrads, axis=(0,1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # Generate heatmap
        (w,h) = (image.shape[2], image.shape[1])
        heatmap = cv.resize(cam.numpy(), (w,h))
        
        # min-max scaling
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap*255).astype('uint8')
        return heatmap
        
    def overlayHeatmap(self, heatmap, image, alpha=0.5, colormap=cv.COLORMAP_TURBO):
        heatmap = cv.applyColorMap(heatmap, colormap)
        output = cv.addWeighted(image, alpha, heatmap, 1-alpha, 0)
        return (heatmap, output)
