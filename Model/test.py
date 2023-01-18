from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from skimage.color import rgb2lab, lab2rgb
from numpy import array, zeros, asarray, uint8
import numpy as np
import sys
import os
from skimage.io import imsave

model = load_model('model.h5')


color_me = []
for filename in os.listdir('./Test/'):
    image = load_img('./Test/' + filename)
    image = image.resize((512,512))
    image_as_array = img_to_array(image)
    image_as_array = np.array(image_as_array, dtype=float)
    color_me.append(image_as_array)

color_me = np.array(color_me, dtype=float)
color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))

# Test model
output = model.predict(color_me)
output = output * 256

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((512, 512, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave("result/img_"+str(i)+".png", lab2rgb(cur))