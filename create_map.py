from gradcam import GradCAM
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import imagenet_utils
import numpy as np
import argparse
import imutils
import cv2 as cv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Create argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
ap.add_argument('-m', '--model', required=True, type=str, default='vgg', 
choices=('vgg', 'resnet'), help='path to model')
args = vars(ap.parse_args())

Model = VGG16 if args['model']=='vgg' else ResNet50
print('loading model...')
model = Model(weights='imagenet')

# Preprocess image
orig = cv.imread(args['image'])
resized = cv.resize(orig, (224,224))
img = load_img(args['image'], target_size=(224,224))
img = img_to_array(img)
img = np.expand_dims(img, axis=0)
img = imagenet_utils.preprocess_input(img)

# Get predictions
preds = model.predict(img)
i = np.argmax(preds[0])
decoded = imagenet_utils.decode_predictions(preds)
(imagenetID, label, prob) = decoded[0][0]
print(f'{label}: {prob*100:.2f}')

# Initialize grad cam and build heatmap
cam = GradCAM(model, i)
heatmap = cam.computeHeatmap(img)
heatmap = cv.resize(heatmap, (orig.shape[1], orig.shape[0]))
# print(type(heatmap))
(heatmap, output) = cam.overlayHeatmap(heatmap, orig, alpha=0.5)
# draw label on output
cv.rectangle(output, (0,0), (340,40), (0,0,0), -1)
cv.putText(output, label, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
output = np.vstack([orig, heatmap, output])
output = imutils.resize(output, height=700)
cv.imshow('output', output)
cv.waitKey(0)
