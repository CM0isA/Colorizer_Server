from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab, gray2rgb
from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
from skimage.transform import resize
import tensorflow as tf
import numpy as np
import os
import random
import gc


# Get images
files = []
# Create batches of 500 images and recursive train on them
for filename in os.listdir('./TrainData/test'):
    files.append(filename)
number_of_batches = int(len(files)/500)
files = np.array(files)
batches = np.array_split(files, number_of_batches + 1)
print("Splitted dataset into", number_of_batches + 1, " batches.")
del number_of_batches
del files
index = 1;

# Generate training data
batches = np.flipud(batches)

# Load resnet weights
inception = InceptionResNetV2(weights='imagenet', include_top=True)
inception.graph = tf.Graph()

for batch in batches:
    print("Batch ", index)
    X=[]
    for filename in batch:
        image = load_img('./TrainData/test/' + filename)
        image = image.resize((512,512))
        image_as_array = img_to_array(image)
        image_as_array = np.array(image_as_array, dtype=float)
        X.append(image_as_array)
        del image
        del image_as_array


    X = np.array(X, dtype=float)
    split = int(0.95*len(X))
    Xtrain = X[:split]
    Xtrain = 1.0/255*Xtrain
    
    # Image transformer
    datagen = ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=20,
            horizontal_flip=True)

    batch_size = 10
    def image_a_b_gen(batch_size):
        for batch in datagen.flow(Xtrain, batch_size=batch_size):
            grayscaled_rgb = gray2rgb(rgb2gray(batch))
            embed = create_inception_embedding(grayscaled_rgb)
            lab_batch = rgb2lab(batch)
            X_batch = lab_batch[:,:,:,0]
            X_batch = X_batch.reshape(X_batch.shape+(1,))
            Y_batch = lab_batch[:,:,:,1:] / 128
            yield ([X_batch, create_inception_embedding(grayscaled_rgb)], Y_batch)

    def create_inception_embedding(grayscaled_rgb):
        grayscaled_rgb_resized = []
        for i in grayscaled_rgb:
            i = resize(i, (299, 299, 3), mode='constant')
            grayscaled_rgb_resized.append(i)
        grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
        grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
        with inception.graph.as_default():
            embed = model.predict(grayscaled_rgb_resized)
        return embed        

    # Train model
    model.fit(image_a_b_gen(batch_size = 10), epochs=5, steps_per_epoch=5)

    # Save model
    model_json = model.to_json()
    with open("resnet.json", "w") as json_file:
        json_file.write(model_json)
    model.save("resnet.h5")


    # Test images
    Xtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]
    Xtest = Xtest.reshape(Xtest.shape+(1,))
    Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]
    Ytest = Ytest / 128
    print("model evaluation ", model.evaluate(Xtest, Ytest, batch_size=batch_size))
    index +=1
    del model
    del Xtrain

