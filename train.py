from mimetypes import init
from pyexpat import model
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np, cv2 as cv, imutils

class GradCAM:
    def __init__(self, model, classIdx, layerName=None) -> None:
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName

        if self.layerName is None:
            self.layerName = self.find_target_layer()
        
